clear;
clc;
close all;
addpath("../");

% immagine da frame video
videoReader = VideoReader("../dataset/mall.mp4");
frameNumber = 0;
while frameNumber < 30
    frame = readFrame(videoReader);
    frameNumber = frameNumber + 1;
end
I = readFrame(videoReader);
imshow(I)
% Queste coordinate in pixel corrispondono a dei punti che nella realtà 
% sappiamo quanto distano => Effettuiamo una corrispondenza metri per pixel
% trasformando la BEV in metri e quindi potendo computare le distanze 
% euclidee tra i punti delle BEV (cioè le distanze tra le persone 
% direttamente in metri)
pts_image = [87 314; 125 247; 502 236; 529 299];

% Distanze in metri corrispondenti ai punti indicati in pts_image
% Considero quindi il punto (0,0) l'angolo in alto a sinistra del rettangolo,
% corrispondentemente all'origine degli assi nella rappresentazione 
% delle immagini in pixel, dell'area di cui conosco le distanze in metri 
pts_world = [0 3; 0 0; 6 0; 6 3];
% Ottengo la trasformazione applicata sull'immagine intera,
% usando come "moving points" i punti del poligono i cui angoli sono i pts_image
% e come "fixed points" i punti in metri all'interno di pts_world
T = fitgeotrans(pts_image, pts_world,'projective');
% Applico la trasformazione T sull'immagine, ottenendo 
% IBird -> immagine trasformata
% RB -> Informazioni sui riferimenti spaziali dell'immagine trasformata
[IBird, RB] = imwarp(I,T);
% Srotolo la matrice Porig per righe, in un vettore riga
point_selected = reshape(pts_world', 1, []);
% Coloro il quadrilatero selezionato di giallo (dopo aver applicato la
% trasformazione
Ori = insertShape(I, 'FilledPolygon', point_selected, 'LineWidth', 5, 'Opacity', 0.5);
imshow(Ori)

%% Scelta del detector
% true = YOLOv4-coco
% false = peopleDetectorACF
yolo = false;

%% Istanzio il PeopleDetector
if yolo == false 
    peopleDetector = peopleDetectorACF();
    [bbox, scores] = detect(peopleDetector, I);

    detectedImg = I;
    detectedImg = utils.getImgPeopleBox(I,bbox);
    imshow(detectedImg)
else
    % Detector YOLOv4
    if exist('net.mat', 'file') == 2
        load net;
        modelName = 'YOLOv4-coco';
    elseif exist('net.mat', 'file') == 0
        modelName = 'YOLOv4-coco';
        model = helper.downloadPretrainedYOLOv4(modelName);
        net = model.net;
        save net net
    end
    
    % Ottengo il nomi delle classi del dataset COCO
    classNames = helper.getCOCOClassNames;

    % Prendo le labels usate per il training del modello preaddestrato
    anchors = helper.getAnchors(modelName);

    % Rilevo gli oggetti nell'immagine
    executionEnvironment = 'auto';
    [bbox, scores, labels] = detectYOLOv4(net, I, anchors, classNames, executionEnvironment);

    % Visualizzo tramite dei box verdi le persone rilevate
    detectedImg = I;
    for i=1:size(bbox,1)
        type = string(labels(i));
        annotation = sprintf('%d', i);
        if type == 'person'
            detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(i,:), annotation, 'LineWidth', 3, 'color', 'green');
        end
    end
    imshow(detectedImg)
end
%%
% Ottengo le coordinate del centro del bounding box (coordinate y al contrario)
bottom_center = [bbox(:,1)+bbox(:,3)/2, bbox(:, 2) + bbox(:,4)];

%% 
[x_world, y_world] = transformPointsForward(T, bottom_center(:,1), bottom_center(:, 2));

%% 
% Calcolo la distanza di una persona dalle altre 
% La matrice 'd' che si ottiene indica la distanza della i-esima 
% dalla j-esima => d è per definizione una matrice simmetrica
distances = pdist2([x_world,y_world], [x_world,y_world]);
% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono superflue)
distances = triu(distances);
% Trovo gli indici delle persone che non rispettano la distanza sociale
% posta a 2 metri
[r, c] = find(distances<2 & distances>0);
idx = [r;c]; 

% =============== BIRD'S EYE VIEW ===============
figure('position',[100 70 1200 600])
% subplot figura a destra (bird's eye view)
sub1 = subplot(1,4,4);
% Segnalo sulla Bird's Eye View tutte le persone indicandole come "Safe"
% Safe -> Colore Blu
plot(x_world, y_world, 'go', 'MarkerFaceColor',"g", 'MarkerSize', 15); hold on
% Sovrascrivo gli indicatori relativi alle persone che non rispettano la 
% distanza sociale con il colore rosso
plot(x_world(idx), y_world(idx), 'ro', 'MarkerFaceColor',"r", 'MarkerSize', 15); 
% Mostro la zona selezionata all'inizio
hold on
pts_world = [pts_world;pts_world(1,:)]
plot(pts_world(:,1), pts_world(:,2), '--b', 'LineWidth',1)
%%
% Setto il rapporto tra assi x,y,z nel grafico
pbaspect([1 2 1])
% Effettuo il reverse dell'asse y per uniformarci col sistema di assi usato
% nelle immagini (origine in alto a sinistra)
set(gca, 'YDir', 'reverse')

% Mostro l'immagine con le persone che non rispettano la distanza sociale
% indicandole con il colore rosso
detectedImg = utils.getImgPeopleBox(detectedImg,bbox,idx);
% subplot figura a sinistra (image with people detected and rectangle)sub2 = subplot(1,4,[1,2,3]);
sub2 = subplot(1,4,[1,2,3]);
imshow(detectedImg)
hold on
% appendo l'ultimo punto per chiudere il poligono
pts_image = [pts_image;pts_image(1,:)];
% plot del corrispondete rettangolo della zona di interesse nell'immagine di partenza
plot(pts_image(:, 1), pts_image(:, 2),'--b','LineWidth',1)
% visualizzo gli assi nell'immagine di partenza
h = gca;
h.Visible = 'On';
axis on;