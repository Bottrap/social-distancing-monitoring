clear;
clc;
close all;

% Recupero l'immagine
% I = imread("../dataset/towncentre.jpg");

videoReader = VideoReader("TownCentreXVID.avi");

frameNumber = 0;
while frameNumber < 30
    frame = readFrame(videoReader);
    frameNumber = frameNumber + 1;
end
I = readFrame(videoReader);

% Parametri del dataset di Oxford
F_X = 2696.35888671875000000000;
F_Y = 2696.35888671875000000000;
C_X = 959.50000000000000000000;
C_Y = 539.50000000000000000000;
S = 0;

% Coefficienti di distorsione (distorsione trascurabile)
radialDistortion =  [-0.60150605440139770508 4.70203733444213867188];
tangentialDistortion = [-0.00047452122089453042 -0.00782289821654558182];

format long;

% Istanzio la matrice dei parametri intrinsechi
intrinsic_matrix = [F_X 0 C_X; 0 F_Y C_Y; 0 0 1];
sz = size(I);
% Rappresentazione di MATLAB della matrice degli intrinsics
% intrinsics = cameraIntrinsics([F_X F_Y], [C_X C_Y], sz(1:2));
% matrix = intrinsics.IntrinsicMatrix;
% params = cameraParameters('IntrinsicMatrix', intrinsic_matrix, 'RadialDistortion', radialDistortion, 'TangentialDistortion', tangentialDistortion);
% J = undistortImage(I,params);
% imshow(J);

% !!!! IN MATLAB IL QUATERNIONE È DEFINITO COME [w x y z]
% !!!! IN PYTHON IL QUATERNIONE È DEFINITO COME [x y z w]
quaternion = [0.49527896681027261394 0.69724917918208628720 -0.43029624469563848566 0.28876888503799524877];

rotation_matrix = quat2rotm(quaternion);

translationVector = [-0.05988363921642303467 3.83331298828125000000 12.39112186431884765625];

cameraMatrix = intrinsic_matrix * [rotation_matrix translationVector'];

% Elimino la colonna relativa all'asse z poichè non mi serve, in quanto i
% punti dell'immagine hanno tutti z = 0
cameraMatrix(:, 3) = [];

P = inv(cameraMatrix);

%% Algoritmo

% Istanzio il PeopleDetector
peopleDetector = peopleDetectorACF();
[bbox, scores] = detect(peopleDetector, I);

detectedImg = I;
detectedImg = utils.getImgPeopleBox(bbox,bbox,detectedImg,'green');
figure
imshow(detectedImg)

% Ottengo le coordinate del centro del bounding box (coordinate y al contrario)
bottom_center = [bbox(:,1)+bbox(:,3)/2, bbox(:, 2) + bbox(:,4)]
% t = insertMarker(detectedImg, bottom_center, 'o', 'size', 40, 'Color', 'r');
% figure, imshow(t)

bottom_center = [bottom_center, ones(size(bottom_center, 1), 1)];

bottom_center_world = [];
for i = 1:size(bottom_center, 1)
    new_row = P * bottom_center(i,:)';
    % Prima era new_row = new_row / new_row(2);
    new_row = new_row / new_row(3);
    bottom_center_world = [bottom_center_world; new_row'];
end

x = bottom_center_world(:, 1);
y = bottom_center_world(:, 2);

% Calcolo la distanza di una persona dalle altre 
% La matrice 'd' che si ottiene indica la distanza della i-esima 
% dalla j-esima => d è per definizione una matrice simmetrica
d = pdist2([x, y], [x, y]);
% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono superflue)
d = triu(d)

% Trovo gli indici delle persone che non rispettano la distanza sociale
% posta a 2 metri
[r, c] = find(d<2 & d>0)
idx = [r;c]; 

%% Visualizzazione Bird's Eye View

imshow(I)
h = drawpolyline('Color','green');
Porig = h.Position;
% Porig = [15.5,529.5;1593.5,847.5;1879.5,239.5;885.5,147.5]
close
% appendo l'ultimo punto per chiudere il poligono
Porig = [Porig;Porig(1,:)];
Porig = [Porig, ones(size(Porig, 1), 1)];
ROI = [];
for i = 1:size(Porig, 1)
    new_row = P * Porig(i,:)';
    % Prima era new_row = new_row / new_row(2);
    new_row = new_row / new_row(3);
    ROI = [ROI; new_row'];
end
ROI_limits = ROI
ROI_limits = utils.rotateMatrix(ROI_limits,-90);

figure('position',[100 70 1200 600]) % default: 0.13 0.11 0.775 0.815
% =============== BIRD'S EYE VIEW ===============
sub1 = subplot(1,4,4);
sub1.Position = sub1.Position + [-0.03 -0.03 0.07 0.07];
plot(ROI_limits(:,1), ROI_limits(:,2), '--b', 'LineWidth',1)
hold on
grid on
% ruoto la matrice
bottom_center_world = utils.rotateMatrix(bottom_center_world,-90);
% plot delle persone nella BEV
plot(bottom_center_world(:,1), bottom_center_world(:,2), 'go','MarkerFaceColor','g', 'MarkerSize',9)
plot(bottom_center_world(idx,1), bottom_center_world(idx,2), 'ro','MarkerFaceColor','r', 'MarkerSize',9)

% image with people detected with red bbox
detectedImg = utils.getImgPeopleBox(idx,bbox,detectedImg,'red');

% image with people detected and rectangle
sub2 = subplot(1,4,[1,2,3]);
sub2.Position = sub2.Position + [-0.03 0 0 0];
imshow(detectedImg)
hold on
plot(Porig(:, 1), Porig(:, 2),'--b','LineWidth',1)
h = gca;
h.Visible = 'On';
axis on;