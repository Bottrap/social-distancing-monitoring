new_BLM = [];
for i=1:size(bodyLocations_meter, 1)
    row = bodyLocations_meter(i,:) / bodyLocations_meter(i,3);
    new_BLM = [new_BLM; row];
end
figure('position',[100 70 1200 600])
plot(new_BLM(:,1), new_BLM(:,2), 'go','MarkerFaceColor','g', 'MarkerSize',9)
%% 
addpath("../");
peopleDetector = peopleDetectorACF();
[bbox, scores] = detect(peopleDetector, I);

distance = 2;
%%
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
% Tramite il people detector e openpose elimino i falsi negativi reciproci
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

%% Tengo (temporaneamente) le colonne dei poses keep
d = d(poses_keep, poses_keep);
%%

% Unione openpose con people detector (red boxes)
d_m = d/1000; %Da mm a m
[r, c] = find(d_m < distance & d_m > 0);
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
detectedImg = I;
detectedImg = utils.getImgPeopleBox(bbox,bbox,detectedImg,'green');
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
        detectedImg = utils.getImgPeopleBox(i, bbox, detectedImg, 'red');
    end
end

imshow(detectedImg);
% poses(
% insidePointIndexes = inpolygon(x_hip, y_hip, x_bbox, y_bbox);

% 
% for i = 1:size(poses,1)
%     for j = 1:size(params.RENDER_PAIRS,1)
%         partA = params.RENDER_PAIRS(j,1);
%         partB = params.RENDER_PAIRS(j,2);
%         
%         x1 = poses(i,partA,1);
%         y1 = poses(i,partA,2);
%         x2 = poses(i,partB,1);
%         y2 = poses(i,partB,2);
%         [x1,y1] = convert_coords(x1,y1,inputHeight,inputWidth,outputHeight,outputWidth);
%         [x2,y2] = convert_coords(x2,y2,inputHeight,inputWidth,outputHeight,outputWidth);
%     end
% end

% Converto un punto dalla dimensione 200x300 (poses) alla dimensione
% effettiva dell'immagine
function [new_X,new_Y] = convert_coords(X,Y,outHeight,outWidth,inHeight,inWidth)
Xratio = double(outHeight) / double(inHeight);
Yratio = double(outWidth) / double(inWidth);
new_X = X * Xratio;
new_Y = Y* Yratio;
end
