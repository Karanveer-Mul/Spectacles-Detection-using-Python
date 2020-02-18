import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from mlxtend.image import extract_face_landmarks

class Face:
    def __init__(self, path):
        self.path = path
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.landmarks = extract_face_landmarks(self.img)
      
        #Individual facial features
        self.nose_mean = np.mean(self.landmarks[[28,29,30]])
        self.mouth_left_corner = self.landmarks[48]
        self.mouth_right_corner = self.landmarks[54]
        self.left_outer_eye = self.landmarks[36]
        self.left_inner_eye = self.landmarks[39]
        self.right_inner_eye = self.landmarks[42]
        self.right_outer_eye = self.landmarks[45] 
        self.left_inner_eyebrow = self.landmarks[21]
        self.right_inner_eyebrow = self.landmarks[22]

    def show_face(self):
        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(1,3,1)
        plt.title("Orignal Image")
        ax.imshow(self.img)
        ax = plt.subplot(1,3,2)
        plt.title("Face landmarks")
        ax.scatter(self.landmarks[:,0],-self.landmarks[:,1])
        ax = plt.subplot(1,3,3)
        plt.title("Landmarks on Orignal Image")
        img2 = self.img.copy()

        for pos in self.landmarks:
            img2[pos[1]-3:pos[1]+3,pos[0]-3:pos[0]+6,:]=(0,0,255) #Assiging blue pixles at position of landmarks
        ax.imshow(img2)
        fig.suptitle("Initial face landmarks detection")
        plt.show()

    def face_normalization(self):
        #We will use trigonometry to find roatation angle required to normalize the face
        h = self.left_outer_eye[1]-self.right_outer_eye[1] #Height of the triangle
        w = self.left_outer_eye[0]-self.right_outer_eye[0] #Width of the triangle
        angle = math.degrees(math.atan(h/w)) #Finding tan inverse and converting to degress for angle of rotation
        image_center = tuple(np.array(self.img.shape[1::-1])/2) 
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0) 
        rotated_img = cv2.warpAffine(self.img, rot_mat, self.img.shape[1::-1], flags=cv2.INTER_LINEAR)
        landmarks = extract_face_landmarks(rotated_img)
        cropped_img = rotated_img[landmarks[24][1]:landmarks[28][1],landmarks[36][0]:landmarks[45][0]] #Area to crop
        self.img = cropped_img

        fig = plt.figure(figsize=(5,4))
        plt.title("Normalised Image")
        plt.imshow(self.img)
        plt.show()
        #Distance from left outer eye to right outer eye
        #eyes_distance_r = math.sqrt((self.left_outer_eye[0]-self.right_outer_eye[0])**2+(self.left_outer_eye[1]-self.right_outer_eye[1])**2)
    
    def edge_detection(self):
        fig = plt.figure(figsize=(10,5))
        edges = cv2.Canny(self.img,200,200)
        h,w = edges.shape
        plt.subplot(1,2,1)
        plt.title("Orignal Image")
        plt.imshow(self.img,cmap='gray')
        plt.subplot(1,2,2)
        plt.title("Edge Image")
        plt.imshow(edges,cmap='gray')
        fig.suptitle("Canny Filter edge detection")
        plt.show()

        #Extracting coordinates from edges
        indices = np.where(edges != [0])
        coordinates =  list(zip(indices[1], indices[0]))

        #Declaring iterator and list
        i = 0
        count = []

        for x in range(int(w*0.45),int(w*0.55)):
            count.append(0)
            for y in range(0,h):
                if (x,y) in coordinates:
                    count[i] += 1
            i += 1
        print(count)

        flag = 1
        for x in count:
            if x<2:
                flag = 0
                print("\nSpectacles not Detected")
                break
        if flag==1:
            print("\nThe image has spectacles")

obj1 = Face('image_path')
obj1.show_face()
obj1.face_normalization()
obj1.edge_detection()
