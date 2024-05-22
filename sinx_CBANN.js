// System configuration:
// Install Node.js
// cd <path>
// npm install mathjs
const math = require('mathjs');
// npm install @tensorflow/tfjs
const tf = require('@tensorflow/tfjs');
// npm install k-medoids
const kmeds = require('k-medoids');
// Execution:
// node <file_name>.js


// Training data
// Function linspace
function linspace(start, end, numPoints) {
    const step = (end - start) / (numPoints - 1);
    const result = [];
    for (let i = 0; i < numPoints; i++) {
        result.push(start + i * step);
    }
    return result;
}
const k = 5;
let x = linspace(0, k * 2 * Math.PI, 500).map(value => [value]);
let y = x.map(value => [(Math.sin(value[0])+1)/2]);
// Query point
const xq = [2];
// Algorithmic settings
// (1) Related to training data:
// Random perturbation tolerance
const pTol = 0.0;
// Scaling factor
const scF = 1000;
// Validation ratio
const tr = 0.1;
// (2) Related to CBANN:
// Number of additional clustering features
const nCl = [2,4,6,8];
// Hidden layer size
const H = Math.pow(2, 8);
// Maximum number of epochs
const max_epochs = 2000;

// Step 1.1: Random perturbation of features
const nVar = x[0].length;
let C = math.dotMultiply(x, math.add(1, math.dotMultiply(pTol, 
    math.random([x.length, nVar], 0, 1))));
let C0 = C;
let C1 = math.concat(C, math.dotMultiply(scF, y), 1);

// Step 1.2: Cluster Boosting (CB)
// Function clusterData
function clusterData(C, k) {
    const clusterer = kmeds.Clusterer.getInstance(C, k);
    const clusteredData = clusterer.getClusteredData();
    const clusterIndices = [];
    for (let i = 0; i < C.length; i++) {
        const observation = C[i];
        let clusterIndex = -1;
        for (let j = 0; j < clusteredData.length; j++) {
            const cluster = clusteredData[j];
            if (cluster.includes(observation)) {
                clusterIndex = j;
                break;
            }
        }
        clusterIndices.push([clusterIndex]);
    }
    return clusterIndices
}
let cInd = [];
for (let i = 0; i < nCl.length; i++) {
    const num_clusters = nCl[i];
    const cInd_i = clusterData(C1, num_clusters);
    cInd.push(cInd_i); 
    C = math.concat(C, cInd_i, 1);
}
C = math.concat(C, math.dotMultiply(scF, y), 1);

// Step 2.1: Divide dataset into training and validation
// Function shuffleArray
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}
// Function selectRandomElements
function selectRandomElements(array, k) {
    const shuffledArray = shuffleArray([...array]);
    return shuffledArray.slice(0, k);
}
const n = C.length;
const nV = Math.round(tr * n);
const indV = selectRandomElements(math.range(0, n - 1)._data, nV);
const indT = math.setDifference(math.range(0, n - 1)._data, indV);
const dataT = math.subset(C, math.index(indT, math.range(0, 
    C[0].length)));
const dataV = math.subset(C, math.index(indV, math.range(0, 
    C[0].length)));
// Extract features and target variable
const y_T = dataT.map(row => [row[row.length - 1]]);
const X_T = dataT.map(row => row.slice(0, row.length - 1));
const y_V = dataV.map(row => [row[row.length - 1]]);
const X_V = dataV.map(row => row.slice(0, row.length - 1));
// Convert input data arrays to TensorFlow tensors
const X_T_tensor = tf.tensor2d(X_T); // training features
const y_T_tensor = tf.tensor2d(y_T); // training labels
const X_V_tensor = tf.tensor2d(X_V); // validation features
const y_V_tensor = tf.tensor2d(y_V); // validation labels

// Step 2.2: Define the neural network architecture and options
const model = tf.sequential();
// Hidden layer with sigmoid activation
model.add(tf.layers.dense({ units: H, activation: 'sigmoid', 
    inputShape: [X_T[0].length] })); 
// Output layer with linear activation
model.add(tf.layers.dense({ units: 1, activation: 'linear' })); 
const adam = tf.train.adam(0.01);
model.compile({ optimizer: adam, loss: 'meanSquaredError' });
// Print model summary
model.summary();

// Step 3: Input vector needed for the ANN for prediction at xq
let ind_d = new Array(nCl.length).fill(0);
for (let i = 0; i < nCl.length; i++) {
    const num_clusters = nCl[i];
    let min_d = new Array(num_clusters).fill(0);
    for (let j = 0; j < num_clusters; j++) {
        const clusterIndices = [];
        for (let index = 0; index < cInd[i].length; index++) {
            if (cInd[i][index][0] === j) {
                clusterIndices.push(index);
            }
        }
        const cp = clusterIndices.map(index => C0[index]);
        const distances = cp.map(row => math.norm(math.subtract(row, xq)));
        min_d[j] = math.min(distances);
    }
    ind_d[i] = min_d.indexOf(Math.min(...min_d));
}
// Formatting of input feature for prediction
const pFeat = math.concat(xq, ind_d);
const pFeat0 = []; 
pFeat0[0] = pFeat;
const pFeat_tensor = tf.tensor2d(pFeat0);

// Step 4: Train the ANN and use it to make prediction on xq
model.fit(X_T_tensor, y_T_tensor, {
    epochs: max_epochs,
    batchSize: 128,
    shuffle: true,
    validationData: [X_V_tensor, y_V_tensor],
    callbacks: {
        onEpochEnd: (epoch, log) => {
          console.log(epoch, log.loss);
        }
    }
    }).then(() => {
console.log("Training complete");
// Prediction results
const pVal0 = model.predict(pFeat_tensor);
const pVal = 1/scF * pVal0.dataSync()[0];
console.log("Input value: ", pFeat);
console.log("Prediction: ", pVal);
console.log("True value: ", (Math.sin(xq)+1)/2);
});
