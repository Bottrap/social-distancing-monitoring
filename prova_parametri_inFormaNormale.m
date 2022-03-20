clear
clc
close all

% Recupero l'immagine
% I = imread("../dataset/towncentre.jpg");

videoReader = VideoReader("./TownCentreXVID.avi");

frameNumber = 0;
while frameNumber < 185
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
prova = false;

%% =================================================
if prova == true
    imshow(I)
    % Suppongo al momento che debba scegliere 3 punti
    % 1) Punto in basso a sinistra
    % 2) Punto in orizzontale al primo punto
    % 3) Punto in verticale al secondo punto
    h = drawpolyline('Color','green');
    Porig = h.Position;
    % load Porig
    % Porig = [15.5,529.5;1593.5,847.5;1879.5,239.5;885.5,147.5]
    close
    bl = [Porig(1,:),1]';
    br = [Porig(2,:),1]';
    t = [Porig(3,:),1]';

    bl = P*bl;
    blw = bl/bl(3);
    br = P*br;
    brw = br/br(3);
    t = P*t;
    tw = t/t(3);
    blw(3) = [];
    brw(3) = [];
    tw(3) = [];
    
    width = sqrt((blw(1)-brw(1))^2 + (blw(2)-brw(2))^2);
    height = sqrt((brw(1)-tw(1))^2 + (brw(2)-tw(2))^2);

    ROI_limits = [blw(1) blw(2); (blw(1)+height) blw(2); (blw(1)+height) (blw(2)-width); blw(1) (blw(2)-width); blw(1) blw(2)];
    % ROI_limits = [blw(1) blw(2); (blw(1)+width) blw(2); blw(1)+width (blw(2)+height); blw(1) (blw(2)+height); blw(1) blw(2)];
else
    % y1 y2 x1 x2
    ROI = [0, 14, 5, 28];
    x1 = 5;
    x2 = 28;
    y1 = 0;
    y2 = 14;
    % Limiti del rettangolo nella BEV (prendo le coppie x1,y1; x2,y2 ecc)
    % Voglio ottenere un rettangolo che abbia dimensione del lato x di 14 metri
    % e del lato y di 23 metri
    ROI_limits = [x1 y1; x1 y2; x2 y2; x2 y1; x1 y1];
end
%% =================================================

% Ottengo i punti corrispondenti, in modo tale da mostrare la regione di
% interesse nell'immagine
pts_camera = [];
for i = 1:size(ROI_limits, 1)
    pt_cam = (cameraMatrix * [ROI_limits(i,1); ROI_limits(i,2); 1])';
    pt_cam = pt_cam / pt_cam(3);
    pts_camera = [pts_camera; pt_cam];
end
pts_camera(:, 3) = [];

% Limiti visualizzazione BEV
BEV_limits = [-20 5 0 30];

figure('position',[100 70 1200 600]) % default: 0.13 0.11 0.775 0.815
% bird's eye view
sub1 = subplot(1,4,4);
sub1.Position = sub1.Position + [-0.03 -0.03 0.07 0.07];
ROI_limits = utils.rotateMatrix(ROI_limits,-90);
plot(ROI_limits(:,1), ROI_limits(:,2), '--b', 'LineWidth',1)
hold on
grid on
axis equal
bottom_center_world = utils.rotateMatrix(bottom_center_world,-90);
plot(bottom_center_world(:,1), bottom_center_world(:,2), 'go','MarkerFaceColor','g', 'MarkerSize',9)
plot(bottom_center_world(idx,1), bottom_center_world(idx,2), 'ro','MarkerFaceColor','r', 'MarkerSize',9)
xlim([BEV_limits(1) BEV_limits(2)]);
ylim([BEV_limits(3) BEV_limits(4)]);

% image with people detected with red bbox
detectedImg = utils.getImgPeopleBox(idx,bbox,detectedImg,'red');

% image with people detected and rectangle
sub2 = subplot(1,4,[1,2,3]);
sub2.Position = sub2.Position + [-0.03 0 0 0];
imshow(detectedImg)
hold on
plot(pts_camera(:, 1), pts_camera(:, 2),'--b','LineWidth',1)
h = gca;
h.Visible = 'On';
axis on;