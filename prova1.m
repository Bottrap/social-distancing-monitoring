%%
I = imread('MatlabImage/cattura2.png');
imshow(I)
h = drawpolyline('Color','green');
Porig = h.Position;
%Porig = [261 209; 719 261; 881 168; 418 125];
sz = size(I);
% Ottengo le coordinate dei 4 angoli dell'immagine
Ppost = [1 1; sz(1) 1; sz(1) sz(2); 1 sz(2)];
% Ottengo la trasformazione applicata sull'immagine intera (Ppost -> "fixed points") 
% e usando come "moving points" i punti del poligono selezionato
T = fitgeotrans(Porig, Ppost,'projective'); 
% Applico la trasformazione T sull'immagine, ottenendo 
% IBird -> immagine trasformata
% RB -> Informazioni sui riferimenti spaziali dell'immagine trasformata
[IBird, RB] = imwarp(I,T);
% Srotolo la matrice Porig per righe, in un vettore riga
point_selected = reshape(Porig', 1, []);
% Coloro il quadrilatero selezionato di giallo (dopo aver applicato la
% trasformazione
Ori = insertShape(I, 'FilledPolygon', point_selected, 'LineWidth', 5, ...
    'Opacity', 0.5);
imshow(Ori)

[x, y] = transformPointsForward(T, Porig(:,1), Porig(:, 2));
[xdataI,ydataI] = worldToIntrinsic(RB,x,y);
% xdataI = x;
% ydataI = y;
IBird2 = insertShape(IBird, 'FilledPolygon', reshape([xdataI, ydataI]', 1, []), "Opacity",0.5);
imshow(IBird2)

%% Istanzio il PeopleDetector
peopleDetector = peopleDetectorACF('caltech-50x21');
[bbox, scores] = detect(peopleDetector, I);

%%
detectedImg = I;
if ~isempty(bbox)
    annotation = sprintf('%s', 'People');
    detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox, annotation, 'LineWidth', 3);
end
imshow(detectedImg)

% Ottengo le coordinate del centro del bounding box (coordinate y al contrario)
bottom_center = [bbox(:,1)+bbox(:,3)/2, bbox(:, 2) + bbox(:,4)];
t = insertMarker(detectedImg, bottom_center, 'o', 'size', 40, 'Color', 'r');
figure, imshow(t)

[x, y] = transformPointsForward(T, bottom_center(:,1), bottom_center(:, 2));
[xdataI,ydataI] = worldToIntrinsic(RB,x,y);
% xdataI = x;
% ydataI = y;
T1 = insertMarker(IBird2, [xdataI, ydataI], "circle", "Size", 20, 'color', 'r');
imshow(T1)

% Calcolo la distanza di una persona dalle altre 
% La matrice 'd' che si ottiene indica la distanza della i-esima 
% dalla j-esima => d è per definizione una matrice simmetrica
d = pdist2([xdataI,ydataI], [xdataI,ydataI]);
% Rendo la matrice ottenuta una matrice triangolare superiore 
% (le informazioni della parte triangolare inferiore sono superflue)
d = triu(d)
% Trovo gli indici delle persone che non rispettano la distanza sociale
% posta a 2 metri
[r, c] = find(d<0.2e3 & d>0)

idx = [r;c]; 
% Segnalo sulla Bird's Eye View tutte le persone indicandole come "Safe"
% Safe -> Colore Blu
plot(xdataI, ydataI, 'bo', 'MarkerFaceColor',"b", 'MarkerSize', 15); hold on
% Sovrascrivo gli indicatori relativi alle persone che non rispettano la 
% distanza sociale con il colore rosso
plot(xdataI(idx), ydataI(idx), 'ro', 'MarkerFaceColor',"r", 'MarkerSize', 15); hold off, axis([size(IBird2, 2)/2 size(IBird2, 2), size(IBird2, 1)/3, size(IBird2, 1)])
for i = 1:length(r)
    line(xdataI([r(i);c(i)]), ydataI([r(i);c(i)]),'color','r')
end
axis equal
% pbaspect([1 2 1])
% Setto il rapporto tra assi x,y,z nel grafico
pbaspect([2 1 1])
% set(gca, 'YDir', 'reverse')
set(gca, 'YDir', 'normal')

% COVID19_SocialDistancing