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
        function detectedImg = getImgPeopleBox(idx,bbox,detectedImg,color)
            if strcmp(color,'green')
                % idx = bbox
                if ~isempty(bbox)
                    for i = 1:size(bbox,1)
                        annotation = sprintf('%d', i);
                        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(i,:), annotation, 'LineWidth', 3, 'color', 'green');
                    end
                end
            elseif strcmp(color,'red')
                if ~isempty(idx)
                    for i = 1:size(idx,1)
                        annotation = sprintf('%d', idx(i));
                        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bbox(idx(i), :), annotation, 'LineWidth', 3, 'color', color); 
                    end
                end
            end
        end
        
    end 
end
