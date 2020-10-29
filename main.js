let model;
let classifierModel;
//let classMap;
async function main() {
    model = await tf.loadLayersModel('decoder-model/model.json');
    classifierModel = await tf.loadLayersModel("conv-model/model.json");
    //classMap = await (await fetch("classificationMap.json")).json();
}
let x = 0;
let y = 0;
const colors = [
    [255, 0, 0],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [255, 125, 0],
    [0, 255, 125],
    [125, 0, 255],
    [125, 255, 0],
];
const legend = document.createElement("div");
legend.innerHTML = "<h3>Legend:</h3>"
colors.forEach((color, i) => {
    legend.innerHTML += `<span style="width:30px;height:30px;background-color:rgb(${color.join(", ")})">&nbsp&nbsp&nbsp&nbsp&nbsp</span> - ${i} ${i % 2 === 0 ? "&nbsp&nbsp&nbsp&nbsp&nbsp": "<br><br>"}`;
})
let classMapImg;

function setup() {
    createCanvas(28 * 20, 28 * 10);
    classMapImg = loadImage("classAreas.png");
    document.body.appendChild(legend);
}

function draw() {
    if (mouseIsPressed) {
        x = constrain(map(mouseX - 28 * 10, 0, 28 * 10, -1, 1), -1, 1);
        y = constrain(map(mouseY, 0, 28 * 10, -1, 1), -1, 1);
    }
    if (model) {
        noStroke();
        const img = numToImg([x, y]);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                fill(img[y * 28 + x]);
                rect(x * 10, y * 10, 10, 10);
            }
        }

    }
    fill(200);
    rect(28 * 10, 0, 28 * 10, 28 * 10);
    image(classMapImg, 280, 0, 280, 280);
    fill(mouseIsPressed ? 100 : 150);
    strokeWeight(5);
    stroke(50);
    circle(map(x, -1, 1, 28 * 10, 28 * 20), map(y, -1, 1, 0, 28 * 10), 25);
}
main();

function numToImg(arr) {
    return model.predict(tf.tensor([arr])).dataSync().map(x => x * 255);
}

function classifyImage(img) {
    const predictions = classifierModel.predict(tf.tensor([img.map(x => x / 255)]).reshape([1, 28, 28, 1])).dataSync();
    const max = Math.max(...predictions);
    return {
        number: predictions.findIndex(pred => pred === max),
        confidence: predictions.find(pred => pred === max)
    }
}