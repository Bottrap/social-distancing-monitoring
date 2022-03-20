%%
clear
clc
close all

videoReader = VideoReader("./mall.mp4");
I = readFrame(videoReader);
% I = imread('cattura1.jpg');
imshow(I)
% Queste coordinate corrispondono a dei punti che nella realtà sappiamo
% quanto distano => Effettuiamo una corrispondenza metri per pixel
% trasformando la BEV in metri e quindi potendo computare le distanze 
% euclidee tra i punti delle BEV (cioè le distanze tra le persone 
% direttamente in metri)
pts_image = [87 314; 125 247; 502 236; 529 299];
% h = drawpolyline('Color','green');
% Porig = h.Position;
% Porig = [261 209; 719 261; 881 168; 418 125];
sz = size(I);
% meter_per_pixel = 0.4 / 30;

% Ottengo le coordinate dei 4 angoli dell'immagine
% Ppost = [1 1; sz(1) 1; sz(1) sz(2); 1 sz(2)];
% pts_world = [0 443.29/2; 0 0; 443.29 0; 443.29 443.29/2] * meter_per_pixel;
% Distanze in metri corrispondenti ai punti indicati prima
% Considero quindi il punto 0,0 quello in alto a sx (corrispondentemente 
% all'origine degli assi nella rappresentazione delle immagini in pixel) 
% dell'area di cui conosco le distanze in metri 
pts_world = [0 3; 0 0; 6 0; 6 3];
% Ottengo la trasformazione applicata sull'immagine intera (Ppost -> "fixed points") 
% e usando come "moving points" i punti del poligono selezionato
T = fitgeotrans(pts_image, pts_world,'projective');
% Applico la trasformazione T sull'immagine, ottenendo 
% IBird -> immagine trasformata
% RB -> Informazioni sui riferimenti spaziali dell'immagine trasformata
[IBird, RB] = imwarp(I,T);
% Srotolo la matrice Porig per righe, in un vettore riga
point_selected = reshape(pts_world', 1, []);
% Coloro il quadrilatero selezionato di giallo (dopo aver applicato la
% trasformazione
Ori = insertShape(I, 'FilledPolygon', point_selected, 'LineWidth', 5, ...
    'Opacity', 0.5);
imshow(Ori)

[x_world, y_world] = transformPointsForward(T, pts_world(:,1), pts_world(:, 2));
IBird2 = insertShape(IBird, 'FilledPolygon', reshape([x_world, y_world]', 1, []), "Opacity",0.5);
imshow(IBird2)

%% Detector choice 
% true = YOLOv4-coco
% false = peopleDetectorACF
yolo = false;

%% Istanzio il PeopleDetector
if yolo == false 
    peopleDetector = peopleDetectorACF();
    [bbox, scores] = detect(peopleDetector, I);

    detectedImg = I;
    if ~isempty(bbox)
        for i = 1:size(bbox,1)
            annotation = sprintf('%d', i);
            detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(i,:), annotation, 'LineWidth', 3, 'color', 'green');
        end
    end
    imshow(detectedImg)
else
    % Prova Detector YOLOv4
    
    load net

    modelName = 'YOLOv4-coco';
    % Get classnames of COCO dataset.
    classNames = helper.getCOCOClassNames;

    % Get anchors used in training of the pretrained model.
    anchors = helper.getAnchors(modelName);

    % Detect objects in test image.
    executionEnvironment = 'auto';
    [bbox, scores, labels] = detectYOLOv4(net, I, anchors, classNames, executionEnvironment);

    % Visualize detection results.
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
d = pdist2([x_world,y_world], [x_world,y_world]);
% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono superflue)
d = triu(d);
% Trovo gli indici delle persone che non rispettano la distanza sociale
% posta a 2 metri
[r, c] = find(d<2 & d>0);

idx = [r;c]; 

% =============== BIRD'S EYE VIEW ===============
figure('position',[100 70 1200 600])
sub1 = subplot(1,4,4);
% Segnalo sulla Bird's Eye View tutte le persone indicandole come "Safe"
% Safe -> Colore Blu
plot(x_world, y_world, 'go', 'MarkerFaceColor',"g", 'MarkerSize', 15); hold on
% Sovrascrivo gli indicatori relativi alle persone che non rispettano la 
% distanza sociale con il colore rosso
plot(x_world(idx), y_world(idx), 'ro', 'MarkerFaceColor',"r", 'MarkerSize', 15); 
%% ZONA DI INTERESSE
hold on
pts_world = [pts_world;pts_world(1,:)]
plot(pts_world(:,1), pts_world(:,2), '--b', 'LineWidth',1)
%%
% Setto il rapporto tra assi x,y,z nel grafico
pbaspect([1 2 1])
% Effettuo il reverse dell'asse y per uniformarci col sistema di assi usato
% nelle immagini (origine in alto a sinistra)
set(gca, 'YDir', 'reverse')

%% image with people detected with red bbox
if ~isempty(idx)
    for i = 1:size(idx,1)
        hold on
        annotation = sprintf('%d', idx(i));
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(idx(i), :), annotation, 'LineWidth', 3, 'color', 'red'); 
    end
end
%%  image with people detected and rectangle
sub2 = subplot(1,4,[1,2,3]);
imshow(detectedImg)
hold on
pts_image = [pts_image;pts_image(1,:)]
plot(pts_image(:, 1), pts_image(:, 2),'--b','LineWidth',1)
h = gca;
h.Visible = 'On';
axis on;

