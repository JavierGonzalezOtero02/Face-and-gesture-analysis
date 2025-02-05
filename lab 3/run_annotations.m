% --- Define the image directory ---
img_path = 'C:\Users\jgojg\Escritorio\4to\2ndotrim\caretos\lacal lab 3\l03_emotion_analysis\group_1'; %ADD YOUR PATH TO THE IMAGES

% --- Get image names and sort them ---
images = dir(fullfile(img_path, '*.jpg'));
imageNames = {images.name}; 
imageNames = erase(imageNames, '.jpg'); 
imageNames = sort(imageNames);  % Ensure correct order

% --- Extract unique emotions and assign sub-indices (1,2,3) ---
uniqueEmotions = unique(cellfun(@(x) strtok(x, '_'), imageNames, 'UniformOutput', false));
formattedLabels = {}; 

for i = 1:length(uniqueEmotions)
    for j = 1:3  % Assuming 3 images per emotion
        formattedLabels{end+1} = sprintf('%s%d', uniqueEmotions{i}, j);
    end
end

% --- Load similarity and consistency matrices ---
data = load('emotions_analysis_v2.mat');
similarityM = data.simScores.similarityM;
consistencyM = data.simScores.consistencyM;

% --- Similarity Heatmap ---
figure;
imagesc(similarityM);
colorbar;
title('Heatmap of SimilarityM');
axis square;
colormap jet;

% Restore axis labels
xticks(1:length(formattedLabels));  
yticks(1:length(formattedLabels));  
xticklabels(formattedLabels);  
yticklabels(formattedLabels);  
xtickangle(45);  % Rotate X labels to avoid overlap

% --- Consistency Heatmap ---
% Replace inf per -1
consistencyM(isinf(consistencyM)) = -1;
figure;
imagesc(consistencyM);
colorbar;
title('Heatmap of ConsistencyM');
axis square;
colormap jet;

xticks(1:length(formattedLabels));  
yticks(1:length(formattedLabels));  
xticklabels(formattedLabels);  
yticklabels(formattedLabels);  
xtickangle(45);

% --- Compute the Dissimilarity Matrix ---
D = zeros(size(similarityM));
for i = 1:size(similarityM, 1)
    for j = 1:size(similarityM, 2)
        D(i,j) = sqrt(similarityM(i,i) - 2*similarityM(i,j) + similarityM(j,j));
    end
end

% --- Dissimilarity Heatmap ---
figure;
imagesc(D);
colorbar;
title('Heatmap of DissimilarityM');
axis square;
colormap jet;

% Restore axis labels
xticks(1:length(formattedLabels));  
yticks(1:length(formattedLabels));  
xticklabels(formattedLabels);  
yticklabels(formattedLabels);  
xtickangle(45);

% --- Multidimensional Scaling (MDS) ---
D_squared = D.^2;
n = size(D_squared, 1);
J = eye(n) - (1/n) * ones(n);  
B = -0.5 * J * D_squared * J;

% Eigenvalue decomposition
[EigVec, EigVal] = eig(B);
[EigValSorted, idx] = sort(diag(EigVal), 'descend'); 
EigVecSorted = EigVec(:, idx);  
X_2D = EigVecSorted(:, 1:2) * sqrt(diag(EigValSorted(1:2)));

% --- Normalize the 2D coordinates ---
X_2D = X_2D - mean(X_2D); % Center the data at (0,0)
X_2D = X_2D / max(abs(X_2D(:))); % Normalize within [-1,1]

% --- Convert to polar coordinates ---
[theta, rho] = cart2pol(X_2D(:,1), X_2D(:,2)); % Convert to (angle, radius)

% --- Scale radius to fit within a unit circle ---
rho = rho / max(rho) * 1; % Ensure all points stay inside the unit circle

% --- Convert back to Cartesian coordinates ---
[X_circle, Y_circle] = pol2cart(theta, rho);

% --- Define Emotion Colors ---
emotionMap = containers.Map({'angry', 'boredom', 'disgusted', 'friendly', 'happiness', 'laughter', 'sadness', 'surprised'}, ...
                            {'r', 'b', 'g', 'y', 'm', 'c', 'k', [1, 0.5, 0.5]});

% --- Extract Emotion Labels ---
emotionLabels = cell(1, length(imageNames));
for i = 1:length(imageNames)
    emotion = strtok(imageNames{i}, '_');
    emotionLabels{i} = emotion;
end

% --- Circular Plot ---
figure;
hold on;

% Draw a thicker reference circle
thetaCircle = linspace(0, 2*pi, 100);
plot(cos(thetaCircle), sin(thetaCircle), 'k', 'LineWidth', 2); % Unit circle

% Draw axes with better visibility
plot([-1 1], [0 0], 'k--', 'LineWidth', 1.5); % Horizontal axis
plot([0 0], [-1 1], 'k--', 'LineWidth', 1.5); % Vertical axis

% Create scatter objects with a better legend structure
legendHandles = [];
legendLabels = {};

for i = 1:length(X_circle)
    if isKey(emotionMap, emotionLabels{i})
        color = emotionMap(emotionLabels{i});
        
        % Add scatter only once per emotion for the legend
        if ~ismember(emotionLabels{i}, legendLabels)
            h = scatter(X_circle(i), Y_circle(i), 100, color, 'filled', 'DisplayName', emotionLabels{i});
            legendHandles = [legendHandles, h];  
            legendLabels{end+1} = emotionLabels{i};
        else
            scatter(X_circle(i), Y_circle(i), 100, color, 'filled');  
        end
    else
        scatter(X_circle(i), Y_circle(i), 100, 'k', 'filled'); % Default to black if emotion is missing
    end
end

% --- Label the quadrants (based on the Circumplex Model) ---
text(0, 1.15, 'Passive/Calm', 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0, -1.15, 'Active/Aroused', 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(1.15, 0, 'Negative', 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 270);
text(-1.15, 0, 'Positive', 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 90);

% --- Add diagonal labels ---
text(0.85, 0.85, 'Low Power/Control', 'FontSize', 12, 'FontAngle', 'italic', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(-0.85, 0.85, 'Conducive', 'FontSize', 12, 'FontAngle', 'italic', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(-0.85, -0.85, 'High Power/Control', 'FontSize', 12, 'FontAngle', 'italic', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0.85, -0.85, 'Obstructive', 'FontSize', 12, 'FontAngle', 'italic', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% --- Adjust Axis Limits and Labels ---
axis equal;
xlim([-1.2 1.2]);
ylim([-1.2 1.2]);

% Add legend in a better location
legend(legendHandles, legendLabels, 'Location', 'eastoutside');

title('Circular Representation of Emotions using MDS');
xlabel('');
ylabel('');

grid on;
hold off;
