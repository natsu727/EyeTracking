import dlib
import cv2
# import pyautogui as pag
import matplotlib.pyplot as plt
import numpy as np

# dlibでフェイストラッキングを行うモデルを読み込む
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# カメラ起動
cap = cv2.VideoCapture(0)

# 瞳の座標を検出する関数
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
	
	# 適用的ヒストグラム平坦化　＝＞　顔を検出しやすい用に画像を明るくする
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# 明るくした画像をモノクロに変換
	cl1 = clahe.apply(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
	# 改めて画像全体を平坦化
	eye = cv2.equalizeHist(cl1)

	eye = eye[org_y:eyes[2].y, org_x:eyes[-1].x]
	_, eye = cv2.threshold(eye, 30, 255, cv2.THRESH_BINARY_INV)
	
	center = get_center(eye)

	avgX = (eyes[0].x +eyes[1].x +eyes[2].x+eyes[3].x)/4
	avgY = (eyes[0].y +eyes[1].y +eyes[2].y+eyes[3].y)/4
	if center:
		return center[0] + org_x, center[1] + org_y , avgX,avgY , eyes[1].y , eyes[2].y
	
	return center , avgX ,avgY , eyes[1].y , eyes[2].y

# 瞳の中心座標を取得する関数
def get_center(gray_img):
	moments = cv2.moments(gray_img, False)
	try:
		#黒目の中心
		return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
	except:
		return 

# 目が閉じていることを確認する関数
def is_close(y0, y1):
	if abs(y0 - y1) < 10:
		return True
	return False

# 瞳座標にポイントを描画する関数
def p(img, parts, eye):
	if eye[0] and None not in eye[0][0:2]:
		cv2.circle(img, eye[0][0:2], 3, (255,255,0), -1)
	if eye[1] and None not in eye[1][0:2]:
		cv2.circle(img, eye[1][0:2], 3, (255,255,0), -1)

	cv2.imshow("me", img)

def is_wink(left_eye,right_eye):

	if left_eye and None not in left_eye[4:6]:
		print("left_eye : "+ str(left_eye[4:6]))
	if right_eye and None not in right_eye[4:6]:
		print("right_eye"+str(right_eye[4:6]))

# キャリブレーション
pos_x=[]
pos_y=[]

# キャリブレーションが完了してるかどうかを示すブール
flame_state=False

# 前5フレーム分の座標を保存する
prev_x=[]
prev_y=[]

while True:

	ret, frame = cap.read()
	frame= cv2.flip(frame,1)
	dets = detector(frame[:,:,::-1])

	if len(dets) > 0:

		parts = predictor(frame, dets[0]).parts()
		left_eye = eye_point(frame, parts)
		right_eye = eye_point(frame, parts, False)
		p(frame, parts, (left_eye, right_eye))
		
		# ここに left_eye と right_eye を引数として動く関数を追加したら良さそう
		# 例：
		#　checkWink(left_eye,right_eye)
		
		is_wink(left_eye,right_eye)
		
		if flame_state and left_eye != None and right_eye != None:

			baseX = (left_eye[2] + right_eye[2])/2
			baseY = (left_eye[3] + right_eye[3])/2
			posX = (left_eye[0]+right_eye[0])/2
			posY = (left_eye[1]+right_eye[1])/2

			if baseX-posX > min(pos_x) and baseX-posX<max(pos_x) and baseY-posY>min(pos_y) and baseY-posY<max(pos_y):
				
				# matplotlibで仮のモニターとしてグラフに枠線描画
				plt.cla()
				minX=min(pos_x)
				minY=min(pos_y)
				MAXX=max(pos_x)
				MAXY=max(pos_y)
				plt.plot([-minX,-MAXX],[minY,minY],color="green")
				plt.plot([-MAXX,-MAXX],[minY,MAXY],color="green")
				plt.plot([-minX,-MAXX],[MAXY,MAXY],color="green")
				plt.plot([-minX,-minX],[minY,MAXY],color="green")
				# plt.plot([-pos_x[0],-pos_x[1]],[pos_y[0],pos_y[1]],color="green")
				# plt.plot([-pos_x[1],-pos_x[2]],[pos_y[1],pos_y[2]],color="green")
				# plt.plot([-pos_x[2],-pos_x[3]],[pos_y[2],pos_y[3]],color="green")
				# plt.plot([-pos_x[3],-pos_x[0]],[pos_y[3],pos_y[0]],color="green")
				
				if len(prev_x) == 5 or len(prev_y) == 5:
					# print("sum(prev_x)/len(prev_x) : ",end="")
					# print(sum(prev_x)/len(prev_x),end="")
					# print(" , sum(prev_y)/len(prev_y) : ",end="")
					# print(sum(prev_y)/len(prev_y))
					# if sum(prev_x)/len(prev_x) > 1.0 or sum(prev_y)/len(prev_y) > 1.0:
						# plt.scatter(baseX-posX,baseY-posY,marker=".")
					# if sum(prev_x)/len(prev_x) >1.5 or sum(prev_y)/len(prev_y) > 0.8:
					
					# 5フレーム分の平均座標を算出しグラフに描画する
					before_avg_x = sum(prev_x)/len(prev_x)
					after_avg_x = sum(prev_x)+baseX-posX/len(prev_x)+1
					before_avg_y = sum(prev_y)/len(prev_y)
					after_avg_y = sum(prev_y)+baseY-posY/len(prev_y)+1





					if before_avg_x-after_avg_x > 0.15 or before_avg_x-after_avg_x < -0.15 or before_avg_y-after_avg_y > 0.15 or before_avg_y-after_avg_y < -0.15 :
						prev_x.append(baseX-posX)
						prev_y.append(baseY-posY)
						plt.scatter(-(sum(prev_x)/len(prev_x)),sum(prev_y)/len(prev_y),marker=".")
						prev_x.pop(0)
						prev_y.pop(0)
					else:
						plt.scatter(-(sum(prev_x)/len(prev_x)),sum(prev_y)/len(prev_y),marker=".")
				else:
					prev_x.append(baseX-posX)
					prev_y.append(baseY-posY)
			
			plt.pause(.01)
	if cv2.waitKey(1) == ord('a'):
		print("左目 : ",end="")
		print(left_eye,end=" ")
		print("右目 : ",end="")
		print(right_eye)
		if not flame_state and left_eye !=None and right_eye !=None and flame_state == False:
			if len(pos_x)<4:
				baseX = (left_eye[2] + right_eye[2])/2
				baseY = (left_eye[3] + right_eye[3])/2
				centerX=(left_eye[0]+right_eye[0])/2
				centerY=(left_eye[1]+right_eye[1])/2
				pos_x.append(baseX-centerX)
				pos_y.append(baseY-centerY)	
				if len(pos_x)==4 :
					 flame_state=True

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
