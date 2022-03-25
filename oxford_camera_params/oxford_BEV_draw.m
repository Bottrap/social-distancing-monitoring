clear;
clc;
close all;
addpath("../");

% immagine da frame video
videoReader = VideoReader("TownCentreXVID.avi");
frameNumber = 0;
while frameNumber < 30
    frame = readFrame(videoReader);
    frameNumber = frameNumber + 1;
end
I = readFrame(videoReader);

% Camera parameters del dataset di Oxford
F_X = 2696.35888671875000000000;
F_Y = 2696.35888671875000000000;
C_X = 959.50000000000000000000;
C_Y = 539.50000000000000000000;
S = 0;

format long;

% Istanzio la matrice dei parametri intrinsechi
intrinsic_matrix = [F_X 0 C_X; 0 F_Y C_Y; 0 0 1];
sz = size(I);

% Istanzio il vettore QUATERNIONE che in Matlab è definito come [w x y z]
quaternion = [0.49527896681027261394 0.69724917918208628720 -0.43029624469563848566 0.28876888503799524877];

% Conversione del quaternione nella matrice di rotazione(3x3)
rotation_matrix = quat2rotm(quaternion);

% Vettore con parametri di traslazione
translationVector = [-0.05988363921642303467 3.83331298828125000000 12.39112186431884765625];

% Calcolo la camera matrix moltiplicando la matrice dei parametri
% intrinseci per una matrice composta da: matrice di rotazione 
% e vettore di traslazione trasposto
cameraMatrix = intrinsic_matrix * [rotation_matrix translationVector'];

% Elimino la colonna relativa all'asse z poichè irrilevante, 
% in quanto i punti dell'immagine hanno tutti z = 0
cameraMatrix(:, 3) = [];

% Calcolo l'inversa della camera matrix al fine di ottenere i world points
P = inv(cameraMatrix);

% Istanzio il PeopleDetector
peopleDetector = peopleDetectorACF();
% Recupero l'output del detector
% bbox <- Posizione degli oggetti rilevati dal detector nell'immagine (Mx4) [x y width height]
% scores <- confidenza del risultato (Mx1)
[bbox, scores] = detect(peopleDetector, I);

% Visualizzo un bounding box verde intorno ad ogni persona rilevata 
% con un'etichetta con il relativo indice
detectedImg = utils.getImgPeopleBox(I,bbox);
% Mostro l'immagine appena modificata 
figure
imshow(detectedImg)

% Ottengo le coordinate del punto centrale del lato inferiore 
% del bounding box [x+width/2, y+height]
bottom_center = [bbox(:,1)+bbox(:,3)/2, bbox(:, 2) + bbox(:,4)]
% Aggiungo la colonna delle cordinate z che sarà un vettore colonna con elementi posti a 1
bottom_center = [bottom_center, ones(size(bottom_center, 1), 1)];

% Istanzio una matrice vuota che verrà riempita con i punti in metri ottenuti dalla trasformazione
bottom_center_world = [];
for i = 1:size(bottom_center, 1)
    % Applico la matrice di trasformazione
    new_row = P * bottom_center(i,:)';
    % Normalizzo ogni riga (x,y,z) dividendo le cordinate x e y per z in
    % modo da avere ogni riga con z pari ad 1
    new_row = new_row / new_row(3);
    % appendo la nuova riga alla matrice
    bottom_center_world = [bottom_center_world; new_row'];
end

% istanzio il vettore con le cordinate x dei punti in metri
x = bottom_center_world(:, 1);
% istanzio il vettore con le cordinate y dei punti in metri
y = bottom_center_world(:, 2);

% Calcolo la distanza di una persona dalle altre 
% La matrice 'distance' che si ottiene indica la distanza della i-esima 
% dalla j-esima ed è per definizione una matrice simmetrica
distance = pdist2([x, y], [x, y]);
% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono superflue)
distance = triu(distance)

% Trovo gli indici delle persone che non rispettano la distanza sociale posta a 2 metri
[r, c] = find(distance<2 & distance>0)
idx = [r;c]; 

%% Visualizzazione Bird's Eye View
imshow(I)
% Traccio un poligono che rappresenta la zona di interesse
h = drawpolyline('Color','green');
% Recupero le cordinate del poligono tracciato
Porig = h.Position; close
% appendo l'ultimo punto per chiudere il poligono
Porig = [Porig;Porig(1,:)];
% Aggiungo la colonna delle cordinate z che sarà un vettore colonna con elementi posti a 1
Porig = [Porig, ones(size(Porig, 1), 1)];

% Istanzio una matrice vuota per i punti della regione di interesse
ROI = [];
for i = 1:size(Porig, 1)
    % Applico la matrice di trasformazione
    new_row = P * Porig(i,:)';
    % Normalizzo ogni riga (x,y,z) dividendo le cordinate x e y per z in
    % modo da avere ogni riga con z pari ad 1
    new_row = new_row / new_row(3);
    % appendo la nuova riga alla matrice
    ROI = [ROI; new_row'];
end
% =============== BIRD'S EYE VIEW ===============
% apro una nuova figura con grandezza e posizione fissata (default: 0.13 0.11 0.775 0.815)
figure('position',[100 70 1200 600])
% subplot figura a destra (bird's eye view)
sub1 = subplot(1,4,4);
sub1.Position = sub1.Position + [-0.03 -0.03 0.07 0.07];
% applico una rotazione di -90° alla matrice con i punti della regione di interesse
ROI_limits = utils.rotateMatrix(ROI,-90);
% plot del rettangolo della zona di interesse nella Bird's Eye View
plot(ROI_limits(:,1), ROI_limits(:,2), '--b', 'LineWidth',1)
hold on; grid on
% applico una rotazione di -90° alla matrice con i punti della regione di interesse
bottom_center_world = utils.rotateMatrix(bottom_center_world,-90);
% plot nella Bird's Eye View di tutte le persone rilevate con un indicatore verde
plot(bottom_center_world(:,1), bottom_center_world(:,2), 'go','MarkerFaceColor','g', 'MarkerSize',9)
% sovrascrivo nella Bird's Eye View tutte le persone che violano la distanza sociale con un indicatore rosso 
plot(bottom_center_world(idx,1), bottom_center_world(idx,2), 'ro','MarkerFaceColor','r', 'MarkerSize',9)

% =============== IMAGE WITH PEOPLE DETECTED ===============
% Visualizzo un bounding box rosso intorno ad ogni persona che viola la distanza sociale
detectedImg = utils.getImgPeopleBox(detectedImg,bbox,idx);
% subplot figura a sinistra (image with people detected and rectangle)sub2 = subplot(1,4,[1,2,3]);
sub2.Position = sub2.Position + [-0.03 0 0 0];
imshow(detectedImg)
hold on
% plot del corrispondete rettangolo della zona di interesse nell'immagine di partenza
plot(Porig(:, 1), Porig(:, 2),'--b','LineWidth',1)
% visualizzo gli assi nell'immagine di partenza
h = gca;
h.Visible = 'On';
axis on;