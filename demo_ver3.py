import dlib
import cv2
import pyautogui as pag
import matplotlib.pyplot as plt
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

def eye_point(img, parts, left=True):
	if left:
		eyes = [
				parts[36],
				min(parts[37], parts[38], key=lambda x: x.y),
				max(parts[40], parts[41], key=lambda x: x.y),
				parts[39],
				]
	else:
		eyes = [
				parts[42],
				min(parts[43], parts[44], key=lambda x: x.y),
				max(parts[46], parts[47], key=lambda x: x.y),
				parts[45],
				]
	org_x = eyes[0].x
	org_y = eyes[1].y
	if is_close(org_y, eyes[2].y):
		return None

	eye = img[org_y:eyes[2].y, org_x:eyes[-1].x]
	_, eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), 30, 255, cv2.THRESH_BINARY_INV)

	center = get_center(eye)

	avgX = (eyes[0].x +eyes[1].x +eyes[2].x+eyes[3].x)/4
	avgY = (eyes[0].y +eyes[1].y +eyes[2].y+eyes[3].y)/4
	if center:
		return center[0] + org_x, center[1] + org_y , avgX,avgY
	return center , avgX ,avgY

def get_center(gray_img):
	moments = cv2.moments(gray_img, False)
	try:
		return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
	except:
		return None

def is_close(y0, y1):
	if abs(y0 - y1) < 10:
		return True
	return False

def p(img, parts, eye):
	if eye[0]:
		cv2.circle(img, (eye[0][0],eye[0][1]), 3, (255, 255, 0), -1)
	if eye[1]:
		cv2.circle(img, (eye[1][0],eye[1][1]), 3, (255, 255, 0), -1)

	cv2.imshow("me", img)
print(pag.size())

while True:
	ret, frame = cap.read()
	dets = detector(frame[:,:,::-1])
	if len(dets) > 0:
		parts = predictor(frame, dets[0]).parts()
		left_eye = eye_point(frame, parts)
		right_eye = eye_point(frame, parts, False)
		p(frame, parts, (left_eye, right_eye))

		if left_eye != None and right_eye != None:
			baseX = (left_eye[2] + right_eye[2])/2
			baseY = (left_eye[3] + right_eye[3])/2
			posX = (left_eye[0]+right_eye[0])/2
			posY = (left_eye[1]+right_eye[1])/2

			plt.plot(baseX-posX,baseY-posY,marker=".")
		plt.pause(.01)
	if cv2.waitKey(1) == ord('a'):
		print("左目 : ",end="")
		print(left_eye,end=" ")
		print("右目 : ",end="")
		print(right_eye)
	
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()