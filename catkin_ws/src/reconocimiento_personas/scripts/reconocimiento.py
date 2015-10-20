#!/usr/bin/env python
# license removed for brevity
import rospy
import roslib
import sys, getopt
import numpy as np
import cv2
import sys
from reconocimiento_personas.msg import coordenadas
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import fnmatch


class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/roah_ipcam/image",Image,self.callback)
    #cnt = 1

  def callback(self,data):
    global cv_image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    (rows,cols,channels) = cv_image.shape

    #cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) # aqui se cambia el color del la ventana de analisis 
    #vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis

def match_and_draw(win, matcher,desc1,desc2,kp1,kp2,img1,img2):
		global fnobj,matches,desc,oportunidades,i,p
        	print 'matching...'
        	raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        	if len(p2) >= 5:
			#rospy.sleep(1.)
        		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                	print '%d / %d  es el objeto ' % (np.sum(status), len(status))
			i = i+1
			if (desc >= 5):
				p = 'b'
				print 'Esta asignando x al logo b'
			else:
				p = 'a'
			oportunidades = oportunidades - 1

        	H, status = None, None
                print '%d caracteristicas encontradas' % len(p1)
		if len(p2) <= 5:
			#option = '/home/labrobotica/hakim/src/face_recognition/scripts/objetos1'
			#rospy.sleep(1.)
			if(desc >= 5):
				#rospy.sleep(5.)
				oportunidades = 0
				for root, dirnames, filenames in os.walk(option):
    					for filename in fnmatch.filter(filenames, 'logo_b*.png'):	
						matches.append(os.path.join(root, filename))
				
				i=0
				fnobj = matches[i]
				print 'Esta buscando el logo b'
				p = 'b'
			desc = desc + 1
				
			oportunidades = oportunidades + 1
			if oportunidades >= 95:
				p = 'c'
			#fnobj = 'cal.png'
			#print 'cambiare de objeto %s ' % fnobj
			
		else:
			vis = explore_match(win, img1, img2, kp_pairs, status, H)

def callback(data):
    global x,y,h,w
    y = data.y
    h = data.h
    x = data.x
    w = data.w
    


def main(args):
  ic = image_converter()
  y=0
  x=0
  h=1
  w=1
  p = 'o'
  u = 0
  rospy.init_node('reconocimiento_final', anonymous=True) 
  rospy.Subscriber("chatter", coordenadas, callback)
  global fnobj, cv_image,cnt,matches,desc,oportunidades,i,option,p
  cnt = 0
  while (y < 0):
  	print 'Aun no llega la activacion para este nodo'
  rate = rospy.Rate(10) #hz
  cap = cv2.VideoCapture(0)

  
  opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
  opts = dict(opts)
  feature_name = opts.get('--feature', 'sift')
  if(cnt == 0):
    print 'Aun no llega la imagen'
  oportunidades = 0
  desc = 0
  i = 0
  matches = []
  option = '/home/labrobotica/objetos1'
  for root, dirnames, filenames in os.walk(option):
    	for filename in fnmatch.filter(filenames, 'logo_a*.png'):	
		matches.append(os.path.join(root, filename))
  fnobj = matches[i]
  while not rospy.is_shutdown():
  #while not oportunidades > 10 | rospy.is_shutdown():
        
    #ret, frame = cap.read()
    crop_img = cv_image[y + 5:h - 5,x + 5:w - 5]   
    cv2.imwrite("full.png",crop_img)
	
	
    img1 = cv2.imread('full.png',0)
    img2 = cv2.imread(fnobj, 0)
    detector, matcher = init_feature(feature_name)
    if detector != None:
      	print 'usando', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))
	
    match_and_draw('analisis', matcher,desc1,desc2,kp1,kp2,img1,img2)
    
    fnobj = matches[i]
    if p == 'o':
	print 'Esta buscando ..., aun no se encuentra nada'
    if p == 'a':
	q = 1 #Es el postman 
	print "Es el postman"
	u = u + 1
    if p == 'b':
	q = 2 #Es el deliman
	print "Es el deliman"
	u = u + 1
    if p == 'c':
	q = 3 #Es la persona desconocida
	print "Es la persona desconocida"
    print 'oprtunidades: ',oportunidades
    if u == 5:
    	sys.exit()
    cv2.waitKey(1)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
