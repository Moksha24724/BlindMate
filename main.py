import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
import numpy as np
import imutils
import time
import cv2
from imutils.video import VideoStream
from threading import Thread, Lock
import os
import requests

#---------------Enter Number-----------

mobile_number = "8431388512"

TRIG = 23
ECHO = 24
VIB = 20
BUTTON = 21


GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(VIB, GPIO.OUT)
GPIO.output(VIB, 1)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

lat = ""
lon = ""

API_KEY = "p7CaYLyQsKuNPZceAIzVGME6Uv18r4g3nwSiF5lWRxTHjq0XtbrWOdHu5e4xET9zA7JmPsBq3DMo6yaI" 

def get_location():
    global lat
    global lon
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5)
        data = res.json()

        lat = data.get("lat")
        lon = data.get("lon")

        if lat and lon:
            return lat, lon
    except Exception as e:
        print("Location Error:", e)

    return None, None

def get_map_link(lat, lon):
    print(f"https://www.google.com/maps?q={lat},{lon}")
    return f"https://www.google.com/maps?q={lat},{lon}"

get_location()

def send_sms(message):
    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "route": "q",
        "message": message,
        "language": "english",
        "numbers": mobile_number
    }

    headers = {
        "authorization": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        print("SMS Response:", response.json())
    except Exception as e:
        print("SMS Error:", e)

def read_distance_cm():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    timeout = start_time + 0.02

    # wait for echo HIGH
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return None
        pulse_start = time.time()

    # wait for echo LOW
    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return None
        pulse_end = time.time()

    duration = pulse_end - pulse_start
    distance_cm = duration * 17150

    return distance_cm


def read_distance_feet():
    d = read_distance_cm()
    if d is None:
        return None
    return d / 30.48

def vibrate_pattern():
    # ON-OFF pattern
    GPIO.output(VIB, False)
    time.sleep(0.3)
    GPIO.output(VIB, True)
    time.sleep(0.2)
    GPIO.output(VIB, False)
    time.sleep(0.3)
    GPIO.output(VIB, True)


def distance_loop():
    last_trigger_time = 0

    while True:
        d_ft = read_distance_feet()

        if d_ft is not None:
            print(f"Distance: {round(d_ft, 2)} ft")

            # trigger vibration only if < 1 ft and not too frequent
            if d_ft < 1:
                if time.time() - last_trigger_time > 2:
                    Thread(target=vibrate_pattern, daemon=True).start()
                    last_trigger_time = time.time()

        time.sleep(0.1)

def button_loop():
    while True:
        if GPIO.input(BUTTON) == 0:  # pressed
            print("Button Pressed")
            if lat and lon:
                link = get_map_link(lat, lon)
                message = f"Emergency! My location: {link}"
            else:
                message = "Emergency! Location unavailable"
            send_sms(message)
            time.sleep(0.3)  # debounce

        time.sleep(0.05)

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("person.cfg", "person.weights")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture(0)
(W, H) = (None, None)

speech_lock = Lock()
last_spoken_time = 0
last_spoken_text = ""
COOLDOWN = 5  # seconds

Thread(target=distance_loop, daemon=True).start()
Thread(target=button_loop, daemon=True).start()

def speak_out(text):
    global last_spoken_time, last_spoken_text

    now = time.time()

    #skip if same text or too soon
    if text == last_spoken_text:
        return
    if now - last_spoken_time < COOLDOWN:
        return

    #skip if already speaking
    if not speech_lock.acquire(blocking=False):
        return

    try:
        os.system(f"espeak-ng '{text}'")
        last_spoken_time = now
        last_spoken_text = text
    finally:
        speech_lock.release()


def audioalert(texts):
    if not texts:
        return

    finaltext = ', '.join(sorted(set(texts)))  # remove duplicates

    t = Thread(target=speak_out, args=(finaltext,))
    t.daemon = True
    t.start()


while True:
	# read the next frame from the file
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
 
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	centers = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
 
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
 
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
 
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				centers.append((centerX, centerY))

				# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	global texts
	texts = []

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#find positions of objects
			centerX, centerY = centers[i][0], centers[i][1]
			if centerX <= W/3:
			    W_pos = "right "
			elif centerX <= (W/3 * 2):
			    W_pos = "center "
			else:
			    W_pos = "left "
					    
			if centerY <= H/3:
			    H_pos = "top "
			elif centerY <= (H/3 * 2):
			    H_pos = "mid "
			else:
			    H_pos = "bottom "

			texts.append(H_pos + W_pos + LABELS[classIDs[i]])

	print(texts)
	cv2.imshow("Image", frame)
	audioalert(texts)

	    
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
			break

# release
print("[INFO] cleaning up...")
vs.stop()
cv2.destroyAllWindows()
GPIO.output(VIB, 1)
GPIO.cleanup()
