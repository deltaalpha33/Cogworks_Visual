import matplotlib.pyplot as plt
from matplotlib import patches as patches
from camera import use_camera
from camera import take_picture

from dlib_models import load_dlib_models
from dlib_models import models
load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]
        
import skimage.io as io
# read a picture in as a numpy-array

import numpy as np

import os

from collections import Counter

class Photo():
    def __init__(self):

        self.upscale = 1

        self.filename = None
        
        self.face_boxes = list()
        
        self.faces = list()
        
    def load_from_file(self, file_path):
        self.file_path = file_path
        self.img_array = io.imread(file_path)
        
    def save_to_file(self, filename = None):
        if filename == None and self.filename == None:
            self.filename = "pls_rename.jpg"
        io.imsave("photos/" + self.filename, self.img_array, plugin=None)
        
    
    def load_from_camera(self):
        with use_camera(port=1, exposure=.5) as camera:
            self.img_array = take_picture()

    def find_faces(self):
        self.detections = face_detect(self.img_array, self.upscale)  # returns sequence of face-detections
        self.detections = list(self.detections)
        
        for det in self.detections:    
            new_face = Face()
            new_face.create_from_photo(self.img_array, det)
            self.faces.append(new_face)
            
    def display_picture(self):
        fig,ax = plt.subplots()
        ax.imshow(self.img_array)
        
    def display_picture_with_boxes(self):
        fig,ax = plt.subplots()
        for face in self.faces:
            ax.add_patch(patches.Rectangle(*face.box, angle=0.0, fill=False, label = face.label))
            ax.text(*face.text_loc, face.label, fontsize=10, color='white')
        ax.imshow(self.img_array)

class Face():
    def __init__(self, descriptor=None, label=""):
        #only used in displaying face from a photo
        self.box = None
        self.text_loc = None
        self.from_photo = False
        
        self.descriptor = descriptor
        
        self.label = label #name
        self.unknown =  True ##true if correct label is not known
        
    def create_from_photo(self, img_data, det):
        self.from_photo = True
        l, r, t, b = det.left(), det.right(), det.top(), det.bottom() 
        self.box = (((r,b), l - r, t-b))
        self.text_loc = (l, b)
        
        
        shape = shape_predictor(img_data, det)
        self.descriptor = np.array(face_rec_model.compute_face_descriptor(img_data, shape))

class Photo_Database():
    def __init__(self,img_dirt = "photo_db", vector_dirt = "vectors"):
        self.database = list() ##array of faces that have been created
        self.img_dirt = img_dirt
        self.vector_dirt = vector_dirt
        self.split = os.sep #splt = \ in windows = / in mac
        
        self.current_photo = None
        
    def load_saved_images(self):
        """
        Loads a db from directory dirt.
        Dirt must be formated like such:
        Folders with names of the desired labels (ie: 'Daschel Cooper')
        Within them .jpg files.
        They will converted to numpy arrays when loaded.
        
        db
        ------
            array of tuples (vector, name)
        """
        
        
        lstOfDirs = [x[0] for x in os.walk(self.img_dirt)][1:] #list of subfolders in directory
        
        
        
        db = []
    
        for rootDir in lstOfDirs:
            print(rootDir)
            fileSet = set()



            for dir_, _, files in os.walk(rootDir): #Loop through all the files
                for fileName in files:
                    relDir = os.path.relpath(dir_, rootDir)
                    relFile = os.path.join(rootDir, fileName)
                    if not fileName.startswith('.'):
                        fileSet.add(relFile)
                        
            for file in fileSet: #Read All the files
                new_photo = Photo()
                new_photo.load_from_file(file)
                new_photo.find_faces()
                name = rootDir.split(self.split)[1]
                for face in new_photo.faces:
                    face.label = name
                    self.database.append(face) #create the description vector and add to database
                    
    def match_face(self, test_face, confidence=0.6):
        """
        Inputs
        ------
        face_descriptor
            vector describing the face
            
        confidence
            how likely it is that it has matched
            
        Returns
        -------
        String(name of face or I do not know)
        """
        dists = list()
        for known_face in self.database:
            dists.append(np.linalg.norm(test_face.descriptor-known_face.descriptor)) #finds the distance between the face_vector and the database vectors
        minimum_distance_index = np.argmin(dists) #finds the index of the vector with the smallest distance
        #print(np.min(dists))
        if(dists[minimum_distance_index] < confidence):
            return self.database[minimum_distance_index].label
        else:
            return "unknown"
    def label_photo(self, photo):
        unknown_label_ext = 0
        for face in photo.faces:
            label = self.match_face(face)
            if label != "unknown":
                face.label = label
                face.unknown = False
            else:
                face.label += label + str(unknown_label_ext)
                unknown_label_ext +=1
    
    def load_photo_from_camera(self):
        current_photo = Photo()
        current_photo.load_from_camera()
        current_photo.find_faces() #must find faces before labeling them!
        self.label_photo(current_photo)
        self.current_photo = current_photo
    
    def input_unknown_labels(self):
        for face in self.current_photo.faces: ##get labels for unkown faces
            if face.unknown == True:
                face.label = input("enter correct name for "  + face.label)
                face.unknown = False
                self.database.append(face) #add labeled face to database
                
    def saveDBnp(self, dirt = "database"):
        """
        Saves a db to directory dirt.
        """
        dirt = os.path.join(os.getcwd() , dirt) ##local working directory
        
        ##sort database
        extension_numbers = Counter() ##counts what the extension should be for each persons face
        for face in self.database:
            direc = dirt + self.split + face.label

            if not os.path.exists(direc):
                os.makedirs(direc)


            direc = direc + self.split +"vct" + str(extension_numbers[face.label])
            extension_numbers[face.label] += 1

            np.save(direc, face.descriptor)
            
    def loadDBnp(self , dirt = "database"):
        """
        Loads a db from directory dirt.
        Dirt must be formated like such:
        Folders with names of the desired labels (ie: 'Daschel Cooper')
        Within them .npz files storing arrays
        """
        dirt = os.path.join(os.getcwd() , dirt) ##local working directory
        
        lstOfDirs = [x[0] for x in os.walk(dirt)][1:]
        self.database = list()
        for rootDir in lstOfDirs:
            #print(rootDir)
            fileList = list()
            for dir_, _, files in os.walk(rootDir):
                for fileName in files:
                    relFile = os.path.join(rootDir, fileName)
                    if not fileName.startswith('.'):
                        fileList.append(relFile)
                    #print(fileName)

            for file in fileList:
                #with load(file) as vector:
                vector = np.load(file)
                name = rootDir.split(self.split)[-1]
                #print(name)
                new_face = Face(descriptor = vector, label = name)
                #print(new_face.descriptor)
                self.database.append( new_face)

