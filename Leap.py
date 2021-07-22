class LEAP_QUARTERNION():
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.v = [self.w, self.x, self.y, self.z] # quarternion as an array

class LEAP_VECTOR():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.v = [self.x, self.y, self.z] # vector as an array

class LEAP_BONE():
    def __init__(self, prev_joint:LEAP_VECTOR, next_joint:LEAP_VECTOR, width, rotation):
        self.prev_joint = prev_joint
        self.next_joint = next_joint
        self.width = width
        self.rotation = rotation

class LEAP_DIGIT():
    def __init__(self, finger_id, is_extended, metacarpal, proximal, intermediate, distal):
        self.finger_id = finger_id
        self.is_extended = is_extended
        self.metacarpal = metacarpal
        self.proximal = proximal
        self.intermediate = intermediate
        self.distal = distal
        self.bones = [self.metacarpal, self.proximal, self.intermediate, self.distal]

class LEAP_PALM():
    def __init__(self, position, stabilized_position, velocity, normal, width, direction, orientation):
        self.position = position
        self.stabilized_position = stabilized_position
        self.velocity = velocity
        self.normal = normal
        self.width = width
        self.direction = direction
        self.orientation = orientation

class LEAP_HAND():
    def __init__(self, id, type, visible_time, pinch_distance, grab_angle, pinch_strength, grab_strength, palm,
                 arm, thumb, index, middle, ring, pinky):
        self.id = id
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

class LEAP_TRACKING_EVENT():
    def __init__(self, frame_id, timestamp, tracking_frame_id, nHands, hands, framerate):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.tracking_frame_id = tracking_frame_id
        self.nHands = nHands
        self.hands = hands
        self.framerate = framerate


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

    return LEAP_DIGIT(finger_id,is_extended,metacarpal,proximal,intermediate,distal)

def get_hand(row):
    hand_id = row['hand_id']
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

    return LEAP_HAND(hand_id, hand_type, visible_time, pinch_distance, grab_angle, pinch_strength,
                     grab_strength, palm, arm, thumb, index, middle, ring, pinky)



def plot_bone(bone:LEAP_BONE, ax):
    prev = bone.prev_joint.v
    next = bone.next_joint.v

    ax.scatter3D(bone.prev_joint.x,bone.prev_joint.y,bone.prev_joint.z, c='r')
    ax.scatter3D(bone.next_joint.x,bone.next_joint.y,bone.next_joint.z, c='r')
    ax.plot3D([bone.prev_joint.x, bone.next_joint.x],
              [bone.prev_joint.y, bone.next_joint.y],
              [bone.prev_joint.z, bone.next_joint.z], c='b')


def plot_digit(digit:LEAP_DIGIT, ax):
    for bone in digit.bones:
        plot_bone(bone,ax)


def plot_digits(hand:LEAP_HAND, ax):
    for digit in hand.digits:
        plot_digit(digit, ax)

def plot_hand(hand:LEAP_HAND, ax):
    plot_digits(hand, ax)
    plot_bone(hand.arm, ax)

    ax.scatter3D(hand.palm.position.x,
                 hand.palm.position.y,
                 hand.palm.position.z, s=[100])

    ax.plot3D([hand.index.metacarpal.next_joint.x, hand.middle.metacarpal.next_joint.x],
              [hand.index.metacarpal.next_joint.y, hand.middle.metacarpal.next_joint.y],
              [hand.index.metacarpal.next_joint.z, hand.middle.metacarpal.next_joint.z], c='b')

    ax.plot3D([hand.middle.metacarpal.next_joint.x, hand.ring.metacarpal.next_joint.x],
              [hand.middle.metacarpal.next_joint.y, hand.ring.metacarpal.next_joint.y],
              [hand.middle.metacarpal.next_joint.z, hand.ring.metacarpal.next_joint.z], c='b')

    ax.plot3D([hand.ring.metacarpal.next_joint.x, hand.pinky.metacarpal.next_joint.x],
              [hand.ring.metacarpal.next_joint.y, hand.pinky.metacarpal.next_joint.y],
              [hand.ring.metacarpal.next_joint.z, hand.pinky.metacarpal.next_joint.z], c='b')