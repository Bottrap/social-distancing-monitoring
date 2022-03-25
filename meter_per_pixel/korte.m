%%
clear
clc
close all
addpath("../");
addpath("../yolov4");

I = imread('../dataset/KORTE/data/_MG_8704.JPG');
imshow(I)
% Seleziono nell'immagine un area della grandezza di 4x4 piastrelle, poichè
% sapendo la grandezza effettiva di una piastrella posso stimare i metri
% per pixel indipendentemente dall'angolatura e della posizione della fotocamera

% Queste coordinate corrispondono a dei punti che nella realtà sappiamo
% quanto distano => Effettuiamo una corrispondenza metri per pixel
% trasformando la BEV in metri e quindi possiamo calcolare le distanze
% euclidee tra i punti delle BEV (cioè le distanze tra le persone
% direttamente in metri)
h = drawpolyline('Color','green');
pts_image = h.Position;
% meter_per_pixel = 0.4 / 30;

% Distanze in metri corrispondenti ai punti selezionati
% Sapendo che ho selezionato un'area pari a 4x4 piastrelle, indico che
% l'area risultante dell'omografia planare è pari a 2.32 x 2.32 metri
% 1 piastrella -> 58 cm
pts_world = [0 0; 2.32 0; 2.32 2.32; 0 2.32];


% Ottengo la trasformazione geometrica applicata ai punti selezionati
% usando come punti fissi quelli indicati con le distanze in metri
T = fitgeotrans(pts_image, pts_world,'projective');
% Applico la trasformazione T sull'immagine, ottenendo
% IBird -> immagine trasformata
% RB -> Informazioni sui riferimenti spaziali dell'immagine trasformata
[IBird, RB] = imwarp(I,T);
% Srotolo la matrice pts_world per righe, in un vettore riga
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
% Ottengo le coordinate del centro del bounding box
bottom_center = [bbox(:,1)+bbox(:,3)/2, bbox(:, 2) + bbox(:,4)];
t = insertMarker(detectedImg, bottom_center, 'o', 'size', 40, 'Color', 'r');
figure, imshow(t)

%%
[x_world, y_world] = transformPointsForward(T, bottom_center(:,1), bottom_center(:, 2));
T1 = insertMarker(IBird2, [x_world, y_world], "circle", "Size", 20, 'color', 'r');
% imshow(T1)

%%
% Calcolo la distanza di una persona dalle altre
% La matrice 'd' che si ottiene indica la distanza della i-esima
% dalla j-esima => d è per definizione una matrice simmetrica
distances = pdist2([x_world,y_world], [x_world,y_world]);
% Rendo la matrice ottenuta una matrice triangolare superiore
% (le informazioni della parte triangolare inferiore sono ridondanti)
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
% Safe -> Colore Verde
plot(x_world, y_world, 'go', 'MarkerFaceColor',"g", 'MarkerSize', 15); hold on
% Sovrascrivo gli indicatori relativi alle persone che non rispettano la
% distanza sociale con il colore rosso
plot(x_world(idx), y_world(idx), 'ro', 'MarkerFaceColor',"r", 'MarkerSize', 15);
% Mostro la zona selezionata all'inizio
hold on
pts_world = [pts_world;pts_world(1,:)];
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
% subplot figura a sinistra
sub2 = subplot(1,4,[1,2,3]);
imshow(detectedImg)
hold on
pts_image = [pts_image;pts_image(1,:)];
% plot del corrispondete rettangolo della zona di interesse nell'immagine di partenza
plot(pts_image(:, 1), pts_image(:, 2),'--b','LineWidth',1)
% visualizzo gli assi nell'immagine di partenza
h = gca;
h.Visible = 'On';
axis on;