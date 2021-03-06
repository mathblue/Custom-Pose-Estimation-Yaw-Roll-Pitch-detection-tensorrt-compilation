import cv2
import math
import numpy as np


def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)
    length=math.sqrt((landmarks[6]+landmarks[0])**2+(landmarks[7]+landmarks[1])**2)
    image_points = np.array([
                            (landmarks[0], landmarks[1]),     # head
                            (landmarks[2], landmarks[3]),     # Left engine
                            (landmarks[4], landmarks[5]),     # Right engine
                            (landmarks[6], landmarks[7])      # bottom engine
                        ], dtype="double")

    

                        
    model_points = np.array([
                            (0.0, length, 0.0),             # head
                            (-(length*14)/53,(length*37)/53, 0.0),     # Left engine
                            ((length*14)/53, (length*37)/53, 0.0),      # Right engine
                            (0.0, 0.0, 0.0),    # bottom engine                        
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[6], landmarks[7]) 

f = open('/Users/Utente/Desktop/poseFINAL/landmark.txt','r')
for line in iter(f):
    img_info = line.split(' ')
    img_path = img_info[0]
    frame = cv2.imread(img_path)
    landmarks =  list(map(int, img_info[1:]))
    
    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)
    
    cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED
    
    #remapping = [2,3,0,4,5,1]
    #for index in range(len(landmarks)/2):
        #random_color = tuple(np.random.random_integers(0,255,size=3))
 
        #cv2.circle(frame, (landmarks[index*2], landmarks[index*2+1]), 5, random_color, -1)  
       # cv2.circle(frame,  tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)  
        
            
#    cv2.putText(frame, rotate_degree[0]+' '+rotate_degree[1]+' '+rotate_degree[2], (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                thickness=2, lineType=2)
                
    for j in range(len(rotate_degree)):
                cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    cv2.imwrite('/Users/Utente/Desktop/poseFINAL/rot.jpg', frame)

f.close()