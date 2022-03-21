sensor_height = 24; %mm
sensor_width = 36; %mm

imgPath = '../dataset/KORTE/data/_MG_8704.JPG';
I = imread(imgPath); 
cameraInfo = imfinfo(imgPath);
focal_length = cameraInfo.DigitalCamera.FocalLength;
realUpperBodyLength = 444.5; %mm

%% PROVIAMO A IMPORTARE OPENPOSE
addpath("utils")

dataDir = fullfile(tempdir,'OpenPose');
% trainedOpenPoseNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/human-pose-estimation.zip';
% downloadTrainedOpenPoseNet(trainedOpenPoseNet_url,dataDir)
% unzip(fullfile(dataDir,'human-pose-estimation.zip'),dataDir);
modelfile = fullfile(dataDir,'human-pose-estimation.onnx');

layers = importONNXLayers(modelfile,"ImportWeights",true);
layers = removeLayers(layers,["Output_node_95" "Output_node_98" "Output_node_147" "Output_node_150"]);
net = dlnetwork(layers);


% The network expects image data of data type single in the range [-0.5, 0.5]. Shift and rescale the data to this range.
netInput = im2single(I)-0.5;
% The network expects the color channels in the order blue, green, red. Switch the order of the image color channels.
netInput = netInput(:,:,[3 2 1]);
% Store the image data as a dlarray.
netInput = dlarray(netInput,"SSC");
% Predict the heatmaps, which are output from the 2-D convolutional layer named 'node_147'.
heatmaps = predict(net,netInput,"Outputs", "node_147");
% Get the numeric heatmap data stored in the dlarray. The data has 19 channels. Each channel corresponds to a heatmap for a unique body part, with one additional heatmap for the background. 
heatmaps = extractdata(heatmaps);
% The OpenPose algorithm does not use the background heatmap to determine the location of body parts. Remove the background heatmap.
heatmaps = heatmaps(:,:,1:end-1);
% Predict the PAFs, which are output from the 2-D convolutional layer named 'node_150'.
pafs = predict(net,netInput,"Outputs","node_150");
% Get the numeric PAF data stored in the dlarray. The data has 38 channels. There are two channels for each type of body part pairing, which represent the x- and y-component of the vector field.
pafs = extractdata(pafs);
%%
params = getBodyPoseParameters;
poses = getBodyPoses(heatmaps,pafs,params);

% Per mostrare le pose
% renderBodyPoses(I,poses,size(heatmaps,1),size(heatmaps,2),params);

%% Mostriamo i numeri delle persone
% imshow(I)
% hold on
% plot(poses(1,2,1), poses(1,2,2), 'go','MarkerFaceColor','g', 'MarkerSize',9)


%% 
% A noi interessano neck-lefthip (7° elemento del vettore)
% e neck-righthip (10° elemento del vettore)

% Grandezza dell'immagine
% image_width = size(I,2);
% image_height = size(I,1);
image_width = 300;
image_height = 200;

bodyLocations = [];

% Considerando che l'openpose e il detector individuino le persone nello
% stesso ordine
for i = 1:size(poses,1)
    x_neck = poses(i, BodyParts.Neck, 1);    
    x_lefthip = poses(i, BodyParts.LeftHip, 1);
    x_righthip = poses(i, BodyParts.RightHip, 1);
    
    x_neck_mm = x_pixelToSensor(x_neck, sensor_width, image_width);
    x_lefthip_mm = x_pixelToSensor(x_lefthip, sensor_width, image_width);
    x_righthip_mm = x_pixelToSensor(x_righthip, sensor_width, image_width);
    
    y_neck = poses(i, BodyParts.Neck, 2);    
    y_lefthip = poses(i, BodyParts.LeftHip, 2);
    y_righthip = poses(i, BodyParts.RightHip, 2);
    
    y_neck_mm = y_pixelToSensor(y_neck, sensor_height, image_height);
    y_lefthip_mm = y_pixelToSensor(y_lefthip, sensor_height, image_height);
    y_righthip_mm = y_pixelToSensor(y_righthip, sensor_height, image_height);
    
    
    dist_neck_lefthip = sqrt((x_neck_mm - x_lefthip_mm)^2 + (y_neck_mm - y_lefthip_mm)^2);
    dist_neck_righthip = sqrt((x_neck_mm - x_righthip_mm)^2 + (y_neck_mm - y_righthip_mm)^2);
    
    % Ci potrebbe essere il caso in cui una delle due distanze stimate tra
    % anca e collo non venga rilevata correttamente => prendiamo
    % direttamente quella di distanza maggiore considerandola come la
    % lunghezza del torso (senza andare a trovare il punto medio tra le
    % anche)
    if dist_neck_lefthip > dist_neck_righthip
        dim_torso = dist_neck_lefthip;
        x_hip_mm = x_lefthip_mm;
        y_hip_mm = y_lefthip_mm;
    else
        dim_torso = dist_neck_righthip;
        x_hip_mm = x_righthip_mm;
        y_hip_mm = y_righthip_mm;
    end
    
    bodySensorXmm = (x_neck_mm + x_hip_mm)/2;
    bodySensorYmm = (y_neck_mm + y_hip_mm)/2;
    
    bodySensorXmm = -(bodySensorXmm - (sensor_width / 2));
    bodySensorYmm = bodySensorYmm - (sensor_height / 2);
    bodySensorZmm = focal_length;
    
    camera_body_ZDistance = 0;
    if dim_torso ~= 0
        camera_body_ZDistance = (focal_length * realUpperBodyLength)/ dim_torso;
    end
    
    % Xa = -(d/f) xa
    bodyRealXmm = -(camera_body_ZDistance / focal_length) * bodySensorXmm;
    % Ya = -(d/f) ya
    bodyRealYmm = -(camera_body_ZDistance / focal_length) * bodySensorYmm;
    % d
    bodyRealZmm = camera_body_ZDistance;
    
    bodyLocations = [bodyLocations; [bodyRealXmm bodyRealYmm bodyRealZmm]];
    
end

d = pdist2([bodyLocations(:,1), bodyLocations(:,2), bodyLocations(:,3)], [bodyLocations(:,1), bodyLocations(:,2), bodyLocations(:,3)]);
d = triu(d);

%% Utilities 

function x_sensor_mm = x_pixelToSensor(x_pixel, sensorWidth, imageWidth)
    x_sensor_mm = (x_pixel * sensorWidth) / imageWidth;
end


function y_sensor_mm = y_pixelToSensor(y_pixel, sensorHeight, imageHeight)
    y_sensor_mm = (y_pixel * sensorHeight) / imageHeight;
end


