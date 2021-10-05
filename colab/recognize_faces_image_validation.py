# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import PIL
from google.colab.patches import cv2_imshow
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--dataset", required=True,
	help="path to input image")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())





# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	name = imagePath.split(os.path.sep)[-2]
  # extract the person name from the image path
	print("[INFO] processing image {}/{} - {}- name {}".format(i + 1,
		len(imagePaths),imagePath,name))

	

	# load the input image and convert it from BGR to RGB
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces..."+args["detection_method"])
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []
	print("[INFO] loop over the facial embeddings...")
	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

			# check to see if we have found a match
		if True in matches:

			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
	print("[INFO] loop over the recognized faces...")
		# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		colorRectangle=(0, 255, 0)		
		if name == "Unknown":
			colorRectangle=(0, 0, 255)		   
		cv2.rectangle(image, (left, top), (right, bottom),colorRectangle, 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, colorRectangle, 2)
	print("[INFO] show the output image...")    
	# show the output image
	cv2_imshow(image)
	#cv2.imshow("Image", image)

	cv2.waitKey(0)


