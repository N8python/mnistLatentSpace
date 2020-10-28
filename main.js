let model;
async function main() {
    model = await tf.loadLayersModel('decoder-model/model.json');
}
let x = 0;
let y = 0;

function setup() {
    createCanvas(28 * 20, 28 * 10);
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
    fill(mouseIsPressed ? 100 : 150);
    strokeWeight(5);
    stroke(50);
    circle(map(x, -1, 1, 28 * 10, 28 * 20), map(y, -1, 1, 0, 28 * 10), 25);
}
main();

function numToImg(arr) {
    return model.predict(tf.tensor([arr])).dataSync().map(x => x * 255);
}