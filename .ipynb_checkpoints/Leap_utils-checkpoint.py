import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import cv2

from tqdm import tqdm
import glob
from PIL import Image


def indent_str(string):
    
    lines = string.split("\n")
    
    output_string = ""
    for line in lines:
        output_string += "\t" + line + "\n"
    
    return output_string

class LEAP_QUARTERNION():
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.v = [self.w, self.x, self.y, self.z]  # quarternion as an array

    def __str__(self):
        str_x = "x: " + str(self.x) + ", "
        str_y = "y: " + str(self.y) + ", "
        str_z = "z: " + str(self.z) + ", "
        str_w = "w: " + str(self.w) + "\n"
        
        return str_x + str_y + str_z + str_w 

class LEAP_VECTOR():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.v = [self.x, self.y, self.z]  # vector as an array
        
    def __str__(self):
        str_x = "x: " + str(self.x) + ", "
        str_y = "y: " + str(self.y) + ", "
        str_z = "z: " + str(self.z) + "\n "
        
        return str_x + str_y + str_z


class LEAP_BONE():
    def __init__(self, prev_joint: LEAP_VECTOR, next_joint: LEAP_VECTOR, width, rotation):
        self.prev_joint = prev_joint
        self.next_joint = next_joint
        self.width = width
        self.rotation = rotation
        
    def __str__(self):
        str_prev_joint = "prev_joint\n" + indent_string(str(self.prev_joint))
        str_next_joint = "next_joint\n" + indent_string(str(self.next_joint))
        str_width = "width: " + str(self.width) + ","
        str_rotation = "rotation: " + str(self.rotation) + "\n"
        
        return str_prev_joint + str_next_joint + str_width + str_rotation
        
class LEAP_DIGIT():
    def __init__(self, finger_id, is_extended, metacarpal, proximal, intermediate, distal):
        self.finger_id = finger_id
        self.is_extended = is_extended
        self.metacarpal = metacarpal
        self.proximal = proximal
        self.intermediate = intermediate
        self.distal = distal
        self.bones = [self.metacarpal, self.proximal, self.intermediate, self.distal]

    
    def __str__(self):
        str_finder_id = str(finger_id) + ", "
        str_is_extended = str(is_extended) + "\n"
        str_metacarpal = "metacarpal\n" + indent_string(str(self.metacarpal))
        str_proximal = "proximal\n" + indent_string(str(self.proximal)) 
        str_intermediate = "intermediate\n" + indent_string(str(self.intermediate)) 
        str_distal= "distal\n" + indent_string(str(self.distal)) 
        
        return str_finger_id + str_is_extended + str_metacarpal + str_intermediate + str_distal
    
class LEAP_PALM():
    def __init__(self, position, stabilized_position, velocity, normal, width, direction, orientation):
        self.position = position
        self.stabilized_position = stabilized_position
        self.velocity = velocity
        self.normal = normal
        self.width = width
        self.direction = direction
        self.orientation = orientation
        
        
    def __str__(self):
        str_position = "position: " + str(self.position) + ", "
        str_stabilized_position = "stabilized position: " + str(self.stabilized_position) + ", "
        str_velocity = "velocity: " + str(self.velocity) + ", "
        str_normal = "normal: " + str(self.normal) + ", "
        str_width = "width:" + str(self.width) + ", "
        str_direction = "direction: " + str(self.direction) + ", "
        str_orientation = "orientation: " + str(self.orientation) + "\n"
        
        return str_position + str_stabilized_position + str_velocity + str_normal + str_width + str_direction + str_orientation


class LEAP_HAND():
    def __init__(self, type, visible_time, pinch_distance, grab_angle, pinch_strength, grab_strength, palm,
                 arm, thumb, index, middle, ring, pinky):
        self.type = type
        self.visible_time = visible_time
        self.pinch_distance = pinch_distance
        self.grab_angle = grab_angle
        self.pinch_strength = pinch_strength
        self.grab_strength = grab_strength
        self.palm = palm
        self.arm = arm
        self.thumb = thumb
        self.index = index
        self.middle = middle
        self.ring = ring
        self.pinky = pinky
        self.digits = [self.thumb, self.index, self.middle, self.ring, self.pinky]
        
    def __str__(self):
        str_type = "type: " + str(self.type) + ", "
        str_visible_time = "visible time: " + str(self.visible_time) + ", "
        str_pinch_distance = "pinch distance: " + str(self.pinch_distance) + ", "
        str_grab_angle = "grab angle: " + str(self.grab_angle) + ", "
        str_pinch_strength = "pinch strength: " + str(self.pinch_strength) + ", "
        str_grab_strength = "grab strength: " + str(self.grab_strength) + ", "
        str_palm = "palm\n" + indent_string(str(self.palm)) + "\n"
        str_arm= "arm\n" + indent_string(str(self.arm)) + "\n"
        str_thumb = "thumb\n" + indent_string(str(self.thumb)) + "\n"
        str_index = "index\n" + indent_string(str(self.index)) + "\n"
        str_middle = "middle\n" + indent_string(str(self.middle)) + "\n"
        str_ring = "ring\n" + indent_string(str(self.ring)) + "\n"
        str_pinky = "pinky\n" + indent_string(str(self.pinky)) + "\n"

        
        return str_type + str_visible_time + str_pinch_distance + str_grab_angle + str_pinch_strength + str_grab_strength + str_palm \
    + str_arm + str_thumb + str_index + str_middle + str_ring + str_pinky
        
class LEAP_TRACKING_EVENT():
    def __init__(self, frame_id, timestamp, tracking_frame_id, nHands, hands, framerate):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.tracking_frame_id = tracking_frame_id
        self.nHands = nHands
        self.hands = hands
        self.framerate = framerate
        
    def __str__(self):
        output_str = ""
        
        output_str += "frame id: " + str(self.frame_id) + ", "
        output_str += "timestamp: " + str(self.timestamp) + ", "
        output_str += "tracking frame id: " + str(self.tracking_frame_id) + ", "
        output_str += "number of hands: " + str(self.nHands) + ", "
        output_str += "framerate: " + str(self.framerate) + "\n"
        for hand in self.hands:
            output_str += indent_str(str(hand)) + "\n"
            
        return output_str
            


def get_palm(row):
    # Palm

    palm_position = LEAP_VECTOR(row["palm_position_x"],
                                row["palm_position_y"],
                                row["palm_position_z"])

    palm_stabilized_position = LEAP_VECTOR(row["palm_stabilized_position_x"],
                                           row["palm_stabilized_position_y"],
                                           row["palm_stabilized_position_z"])

    palm_velocity = LEAP_VECTOR(row["palm_velocity_x"],
                                row["palm_velocity_y"],
                                row["palm_velocity_z"])

    palm_normal = LEAP_VECTOR(row["palm_normal_x"],
                              row["palm_normal_y"],
                              row["palm_normal_z"])

    palm_width = row["palm_width"]

    palm_direction = LEAP_VECTOR(row["palm_direction_x"],
                                 row["palm_direction_y"],
                                 row["palm_direction_z"])

    palm_orientation = LEAP_QUARTERNION(row["palm_orientation_w"],
                                        row["palm_orientation_x"],
                                        row["palm_orientation_y"],
                                        row["palm_orientation_z"], )

    return LEAP_PALM(palm_position, palm_stabilized_position, palm_velocity, palm_normal,
                     palm_width, palm_direction, palm_orientation)


def get_bone(row, name):
    prev_joint = LEAP_VECTOR(row[name + "_prev_joint_x"],
                             row[name + "_prev_joint_y"],
                             row[name + "_prev_joint_z"])

    next_joint = LEAP_VECTOR(row[name + "_next_joint_x"],
                             row[name + "_next_joint_y"],
                             row[name + "_next_joint_z"])

    width = row[name + "_width"]

    rotation = LEAP_QUARTERNION(row[name + "_roatation_w"],
                                row[name + "_roatation_x"],
                                row[name + "_roatation_y"],
                                row[name + "_roatation_z"])

    return LEAP_BONE(prev_joint, next_joint, width, rotation)


def get_digit(row, name):
    finger_id = row[name + "_finger_id"]
    is_extended = row[name + "_is_extended"]

    metacarpal = get_bone(row, name + "_metacarpal")
    proximal = get_bone(row, name + "_proximal")
    intermediate = get_bone(row, name + "_intermediate")
    distal = get_bone(row, name + "_distal")

    return LEAP_DIGIT(finger_id, is_extended, metacarpal, proximal, intermediate, distal)


def get_hand(row):
    hand_type = row['hand_type']
    visible_time = row["visible_time"]
    pinch_distance = row["pinch_distance"]
    grab_angle = row["grab_angle"]
    pinch_strength = row["pinch_strength"]
    grab_strength = row["grab_strength"]

    palm = get_palm(row)

    arm = get_bone(row, "arm")
    thumb = get_digit(row, "thumb")
    index = get_digit(row, "index")
    middle = get_digit(row, "middle")
    ring = get_digit(row, "ring")
    pinky = get_digit(row, "pinky")

    return LEAP_HAND(hand_type, visible_time, pinch_distance, grab_angle, pinch_strength,
                     grab_strength, palm, arm, thumb, index, middle, ring, pinky)

def plot_hand(hand: LEAP_HAND, ax):
    
    def plot_bone(bone: LEAP_BONE, ax):
        prev_bone = bone.prev_joint.v
        next_bone = bone.next_joint.v

        ax.scatter3D(bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z, c='r')
        ax.scatter3D(bone.next_joint.x, bone.next_joint.y, bone.next_joint.z, c='r')
        ax.plot3D([bone.prev_joint.x, bone.next_joint.x],
                  [bone.prev_joint.y, bone.next_joint.y],
                  [bone.prev_joint.z, bone.next_joint.z], c='b')
    
    # Plotting digits and arm
    
    for digit in hand.digits:
         for bone in digit.bones:
                plot_bone(bone)
    
    
    plot_bone(hand.arm, ax)
    
    
    # Making the palm position more visible than others

    ax.scatter3D(hand.palm.position.x,
                 hand.palm.position.y,
                 hand.palm.position.z, s=[100])

    # Ploting outline of the hand
    
    ax.plot3D([hand.index.metacarpal.next_joint.x, hand.middle.metacarpal.next_joint.x],
              [hand.index.metacarpal.next_joint.y, hand.middle.metacarpal.next_joint.y],
              [hand.index.metacarpal.next_joint.z, hand.middle.metacarpal.next_joint.z], c='b')

    ax.plot3D([hand.middle.metacarpal.next_joint.x, hand.ring.metacarpal.next_joint.x],
              [hand.middle.metacarpal.next_joint.y, hand.ring.metacarpal.next_joint.y],
              [hand.middle.metacarpal.next_joint.z, hand.ring.metacarpal.next_joint.z], c='b')

    ax.plot3D([hand.ring.metacarpal.next_joint.x, hand.pinky.metacarpal.next_joint.x],
              [hand.ring.metacarpal.next_joint.y, hand.pinky.metacarpal.next_joint.y],
              [hand.ring.metacarpal.next_joint.z, hand.pinky.metacarpal.next_joint.z], c='b')


if __name__ == "__main__":

    example_path = r"E:\Ruijin\Recording\20210621-132108_songshanying"

    recording_id = example_path.split("\\")[-1]

    leap_list = glob.glob(example_path + r"\*_leap.csv")

    leap_recordings = {}
    for x in leap_list:
        rm_path = x.replace(example_path + "\\", "")
        rm_id = rm_path.replace(recording_id + "_", "")
        leap_df = pd.read_csv(x)

    # file = r"C:\Users\Cam\Downloads\Ruijin_test\20210529-092302_z963014\20210529-092302_z963014_20210529-093406_visual_leap.csv"

    tracking_events = []

    # Current tracking frame
    prev_tracking_frame = None

    temp_hand = None

    for index, row in leap_df.iterrows():

        prev_tracking_frame = row['frame_id']

        frame_id = row['frame_id']
        timestamp = row['timestamp']
        tracking_frame_id = row['tracking_frame_id']
        nHands = row['nHands']
        framerate = row['framerate']
        hand = get_hand(row)

        if nHands == 1:
            tracking_events.append(LEAP_TRACKING_EVENT(frame_id, timestamp, tracking_frame_id, nHands, [hand],
                                                       framerate))
        else:
            if row['frame_id'] == prev_tracking_frame:
                # There has been another hand tracked
                tracking_events.append(LEAP_TRACKING_EVENT(frame_id, timestamp, tracking_frame_id, nHands,
                                                           [temp_hand, hand], framerate))
            else:
                temp_hand = hand

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 120.0, (500, 400))

    count = 0

    for event in tqdm(tracking_events):

        # # make a Figure and attach it to a canvas.
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)

        fig = plt.figure(figsize=(5, 4), dpi=100)
        # canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-150, 150)
        ax.set_ylim3d(-150, 150)
        ax.set_zlim3d(-150, 150)

        ax.view_init(125, -140)

        #
        plt.axis('off')
        #
        for hand in event.hands:
            plot_hand(hand, ax)
        #
        #     # Retrieve a view on the renderer buffer
        #     canvas.draw()
        #     buf = canvas.buffer_rgba()
        #     # convert to a NumPy array
        #     X = np.asarray(buf)
        #
        #     img = Image.fromarray(X.astype(np.uint8))
        #     opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #     #print(opencvImage.shape)
        #     #cv2.imshow("hello", opencvImage)
        #     #cv2.waitKey(0)
        #
        #     out.write(opencvImage)
        #     plt.close(fig)
        #
        #
        # out.release()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # example_event = tracking_events[10]
        # example_hand = example_event.hands[0]
        # plot_hand(example_hand, ax)

        plt.show()

        break
