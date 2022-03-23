% Represent body parts as enum with integer value so they can be used as
% indices
classdef BodyParts < uint8
    % Using COCO indices for body parts
    enumeration
        Nose(1)
        Neck(2)
        LeftShoulder(3)
        LeftElbow(4)
        LeftHand(5)
        RightShoulder(6)
        RightElbow(7)
        RightHand(8)
        LeftHip(9)
        LeftKnee(10)
        LeftFoot(11)
        RightHip(12)
        RightKnee(13)
        RightFoot(14)
        LeftEye(15)
        RightEye(16)
        LeftEar(17)
        RightEar(18)
    end
end