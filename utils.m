classdef utils
    methods(Static)
        %% rotate matrix
        function matrixRotated = rotateMatrix(matrix,angle)
            % matrix: matrice 
            if size(matrix,2) == 3
                % rimuovo l'asse z
                matrix(:,3) = [];
            end
            % angles in rad
            angle = deg2rad(angle);
            % definisco matrice di rotazione per visualizzare correttamente
            rotation = [cos(angle) -sin(angle);sin(angle) cos(angle)];
            % ruoto la matrice
            matrixRotated = matrix*rotation;
        end
        
        %% get image with red bbox on people detected 
        function detectedImg = getImgPeopleRedBox(idx,bbox,detectedImg)
            if ~isempty(idx)
                for i = 1:size(idx,1)
                    annotation = sprintf('%d', idx(i));
                    detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(idx(i), :), annotation, 'LineWidth', 3, 'color', 'red'); 
                end
            end
        end
        
    end 
end
