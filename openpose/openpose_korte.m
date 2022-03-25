sensor_height = 24; %mm
sensor_width = 36; %mm

addpath("../");

% Definisco il percorso dell'immagine che voglio importare
imgPath = '../dataset/KORTE/data/_MG_8781.JPG';
I = imread(imgPath); 
cameraInfo = imfinfo(imgPath);
focal_length = cameraInfo.DigitalCamera.FocalLength;
realUpperBodyLength = 444.5; %mm

%% Importo Openpose
addpath("utils")

dataDir = fullfile(tempdir,'OpenPose');
trainedOpenPoseNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/human-pose-estimation.zip';
downloadTrainedOpenPoseNet(trainedOpenPoseNet_url,dataDir)
unzip(fullfile(dataDir,'human-pose-estimation.zip'),dataDir);
modelfile = fullfile(dataDir,'human-pose-estimation.onnx');

layers = importONNXLayers(modelfile,"ImportWeights",true);
layers = removeLayers(layers,["Output_node_95" "Output_node_98" "Output_node_147" "Output_node_150"]);
net = dlnetwork(layers);


%% OpenPose detection (https://it.mathworks.com/help/deeplearning/ug/estimate-body-pose-using-deep-learning.html)
% oppure usare il comando --> openExample('deeplearning_shared/EstimateBodyPoseUsingDeepLearningExample')

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
renderBodyPoses(I,poses,size(heatmaps,1),size(heatmaps,2),params);

%% Istanzio il peopleDetector
peopleDetector = peopleDetectorACF();
[bbox, scores] = detect(peopleDetector, I);

social_distance = 2;

%%
% A noi interessano neck-lefthip (7° elemento del vettore)
% e neck-righthip (10° elemento del vettore)

% Considero come grandezza dell'immagine quella scalata e utilizzata da
% openpose
image_width = 300;
image_height = 200;

bodyLocations = [];

%Ottengo le coordinate di collo e anche delle persone rilevate da Openpose
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

% Calcolo la distanza di una persona dalle altre 
% La matrice 'd' che si ottiene indica la distanza della i-esima 
% dalla j-esima => d è per definizione una matrice simmetrica
distances = pdist2([bodyLocations(:,1), bodyLocations(:,2), bodyLocations(:,3)], [bodyLocations(:,1), bodyLocations(:,2), bodyLocations(:,3)]);

% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono ridondanti)
distances = triu(distances);

%% Visualizzazione box
% Prendo le coordinate relative ai punti di una sola anca per ogni persona
% Nel caso in cui sia LeftHip che RightHip siano NaN, pongo la riga pari a
% 0 e poi andrò ad eliminare la persona da poses
hips_points = [];
for i = 1:size(bbox, 1)
    if ~isnan(poses((i), BodyParts.LeftHip, 1)) && ~isnan(poses((i), BodyParts.LeftHip, 2))
        x_hip = poses(i, BodyParts.LeftHip, 1);
        y_hip = poses(i, BodyParts.LeftHip, 2);
        [new_x_hip, new_y_hip] = convert_coords(x_hip, y_hip, 1600, 2400, 200, 300);
        hips_points = [hips_points; new_x_hip new_y_hip];
    elseif ~isnan(poses((i), BodyParts.RightHip, 1)) && ~isnan(poses((i), BodyParts.RightHip, 2))
         x_hip = poses(i, BodyParts.RightHip, 1);
         y_hip = poses(i, BodyParts.RightHip, 2);
        [new_x_hip, new_y_hip] = convert_coords(x_hip, y_hip, 1600, 2400, 200, 300);
        hips_points = [hips_points; new_x_hip new_y_hip];
    else 
        hips_points = [hips_points; 0 0];
    end
end
% Tramite il people detector e openpose elimino i falsi positivi reciproci
bbox_keep = [];
poses_keep = [];
for i = 1:size(bbox,1)
    bbox_points = bbox2points(bbox(i,:));
    x_box = bbox_points(:, 1);
    y_box = bbox_points(:, 2);
    indexes = inpolygon(hips_points(:,1), hips_points(:,2), x_box, y_box);
    if any(indexes, 'all') == 1
        % Mi salvo gli indici dei bbox e dei punti di poses che sono corretti
        bbox_keep = [bbox_keep; i];
        poses_keep = [poses_keep; find(indexes == 1)];
    end  
end

%% Rimuovo i punti e i bbox che mi danno un falso positivo
bbox = bbox(bbox_keep, :);
poses = poses(poses_keep, :, :);

%% Tengo le colonne dei poses keep
distances = distances(poses_keep, poses_keep);
%%

% Unione openpose con people detector (red boxes)
% Esprimo la matrice delle distanze in metri
d_m = distances/1000; %Da mm a m
[r, c] = find(d_m < social_distance & d_m > 0);
idx = [r;c];
idx = unique(idx);

% Ottengo le coordinate delle anche delle persone che violano la distanza
% sociale e le trasformo in relazione alla dimensione effettiva
% dell'immagine e non rispetto a 200x300 (dim. scalata per la openpose)
viol_hip = [];
for i = 1:size(idx)
    if ~isnan(poses(idx(i), BodyParts.LeftHip, 1)) && ~isnan(poses(idx(i), BodyParts.LeftHip, 2))
        x_hip = poses(idx(i), BodyParts.LeftHip, 1);
        y_hip = poses(idx(i), BodyParts.LeftHip, 2);
        [new_x_hip, new_y_hip] = convert_coords(x_hip, y_hip, 1600, 2400, 200, 300);
        viol_hip = [viol_hip; new_x_hip, new_y_hip];
    end
    if ~isnan(poses(idx(i), BodyParts.RightHip, 1)) && ~isnan(poses(idx(i), BodyParts.RightHip, 2))
        x_hip = poses(idx(i), BodyParts.RightHip, 1);
        y_hip = poses(idx(i), BodyParts.RightHip, 2);
        [new_x_hip, new_y_hip] = convert_coords(x_hip, y_hip, 1600, 2400, 200, 300);
        viol_hip = [viol_hip; new_x_hip, new_y_hip];
    end
end

% Mostro l'immagine dopo aver rimosso i falsi positivi
detectedImg = utils.getImgPeopleBox(I,bbox);
figure
imshow(detectedImg)

% Faccio mostrare in rosso il bbox relativo alle persone che violano la
% distanza sociale
for i = 1:size(bbox,1)
    bbox_points = bbox2points(bbox(i,:));
    x_box = bbox_points(:, 1);
    y_box = bbox_points(:, 2);
    indexes = inpolygon(viol_hip(:,1), viol_hip(:,2), x_box, y_box);
    % Controllo che indexes abbia almeno un elemento pari a 1
    if any(indexes, 'all') == 1
        detectedImg = utils.getImgPeopleBox(detectedImg,bbox,idx);
    end
end

imshow(detectedImg);


%% Utilities 

function x_sensor_mm = x_pixelToSensor(x_pixel, sensorWidth, imageWidth)
    x_sensor_mm = (x_pixel * sensorWidth) / imageWidth;
end


function y_sensor_mm = y_pixelToSensor(y_pixel, sensorHeight, imageHeight)
    y_sensor_mm = (y_pixel * sensorHeight) / imageHeight;
end

% Converto un punto dalla dimensione 200x300 (poses) alla dimensione
% effettiva dell'immagine
function [new_X,new_Y] = convert_coords(X,Y,outHeight,outWidth,inHeight,inWidth)
Xratio = double(outHeight) / double(inHeight);
Yratio = double(outWidth) / double(inWidth);
new_X = X * Xratio;
new_Y = Y* Yratio;
end





