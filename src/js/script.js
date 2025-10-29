import getReadyImg from '../assets/img/Get-Ready-To-Memorize.svg';
import flatHandMemorizeImg from '../assets/img/Flat-Hand-Memorize.svg';
import peaceMemorizeImg from '../assets/img/Peace-Memorize.svg';
import thumbsUpMemorizeImg from '../assets/img/Thumbs-Up-Memorize.svg';
import blackMemorizeImg from '../assets/img/Black-Memorize.svg';
import handGameImg from '../assets/img/Hand-Game.svg';
import peaceGameImg from '../assets/img/Peace-Game.svg';
import thumbsUpGameImg from '../assets/img/Thumbs-Up-Game.svg';
import handGameOverImg from '../assets/img/Hand-Game-Over.svg';
import peaceGameOverImg from '../assets/img/Peace-Game-Over.svg';
import thumbsUpGameOverImg from '../assets/img/Thumbs-Up-Game-Over.svg';
import circleEmpty from '../assets/img/Circle-Empty.svg';
import circleGreen from '../assets/img/Circle-Green.svg';
import circleRed from '../assets/img/Circle-Red.svg';
import circleFilled from '../assets/img/Circle-Filled.svg';

import correctSound from '../assets/sounds/correct.ogg';
import wrongSound from '../assets/sounds/wrong.ogg';

import loadingVideo from '../assets/vid/loading.mp4';
document.querySelector('#loading-load source').setAttribute('src', loadingVideo);
document.querySelector('#train-load source').setAttribute('src', loadingVideo);

const modelJSON = new URL('/model/model.json', import.meta.url).href;
const modelWeights = new URL('/model/model.weights.bin', import.meta.url).href;

const $state = document.querySelector('#state');
const $canvas = document.querySelector('#canvas');
const $flatHand = document.querySelector('#flat-hand');
const $peace = document.querySelector('#peace');
const $thumbsUp = document.querySelector('#thumbs-up');
const $train = document.querySelector('#train');
const $save = document.querySelector('#save');
const $sampelingHome = document.querySelector('#sampling-home');

const $training = document.querySelector('#training');
const $results = document.querySelector('#results');

const $homeStart = document.querySelector('#home-start');
const $create = document.querySelector('#create');

const $infoBack = document.querySelector('#info-back');
const $start = document.querySelector('#start');

const $gameOverHome = document.querySelector('#game-over-home');
const $playAgain = document.querySelector('#play-again');

let video, ctx;
let handPose;
let classifier;
let hands = [];
let classificationResults = [];

const signs = ["flat-hand", "peace", "thumbs-up"];
let signPattern = [];
let rightPattern = [];
let round = 0;
let signOfRound = 0;
let gameStage = "memorizeStage";
let memorizeSign = -1;

let flatHandSamples = 0;
let thumbsUpSamples = 0;
let peaceSamples = 0;

const STATE_LOADING = "loading";
const STATE_INTRO = "intro";
const STATE_INFO = "info";
const STATE_GAME = "game";
const STATE_GAME_OVER = "game-over";
const STATE_SAMPLING = "sampling";
const STATE_TRAINING = "training";
const ALL_STATES = [
    STATE_LOADING,
    STATE_INTRO,
    STATE_INFO,
    STATE_GAME,
    STATE_GAME_OVER,
    STATE_SAMPLING,
    STATE_TRAINING,
];
let state = STATE_LOADING;

//inf because better then nothing
let timeToCheck = Number.POSITIVE_INFINITY;
let timeToShowMemorize = Number.POSITIVE_INFINITY;

const setState = (value) => {
    state = value;
    $state.textContent = state;
    document.documentElement.classList.remove(...ALL_STATES);
    document.documentElement.classList.add(state);
    //sets the timers
    document.querySelectorAll('.display-game, .cam__ontop').forEach(el => el.classList.add('hidden'));
    if (value === STATE_GAME) {
        document.querySelector('.display-game').classList.remove('hidden');
        timeToCheck = performance.now() + 3000;
        timeToShowMemorize = performance.now() + 2000;
    }
    if (value === STATE_SAMPLING) {
        document.querySelector('.display-sampling').classList.remove('hidden');
        document.querySelector(`.game-bottom`).style.opacity = 0;
    }
};

const preload = async () => {
    setState(STATE_LOADING);
    requestAnimationFrame(draw);
    handPose = ml5.handPose();
    await handPose.ready;
    setup();
}

const gameReset = () => {
    rightPattern.forEach(sign => {
        document.querySelector(`.progress-bar`).removeChild(document.querySelector(`.progress-bar`).lastElementChild)
    });

    signPattern = [];
    rightPattern = [];
    round = 0;
    signOfRound = 0;
    gameStage = "memorizeStage";
    memorizeSign = -1;
}

const setup = async () => {
    drawLoading();
    ctx = $canvas.getContext('2d');
    // create a video stream - specify a fixed size
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: 440,
            height: 300
        }
    });
    video = document.createElement('video');
    video.srcObject = stream;
    video.play();
    // set canvas & video size
    $canvas.width = video.width = 440;
    $canvas.height = video.height = 300;
    // start detecting hands
    handPose.detectStart(video, (results) => {
        hands = results;
        if (state === STATE_GAME) {
            if (hands.length === 0) {
                return;
            }
            // the keypoints should be one big array of numbers
            const keypoints = getRelevantPositionKeyPoints(hands[0]).map(keypoint => [keypoint.x, keypoint.y]).flat();
            classifier.classify(keypoints, (results) => {
                classificationResults = results;
            });
        }
    });
    // For this example to work across all browsers
    // "webgl" or "cpu" needs to be set as the backend
    ml5.setBackend("webgl");
    // Set up the neural network
    let classifierOptions = {
        inputs: 42,
        outputs: 3,
        task: "classification",
        debug: true,
    };
    
    classifier = ml5.neuralNetwork(classifierOptions);
    const pretrainedModelURL = modelJSON;
    // add event listeners to buttons
    $flatHand.addEventListener('click', () => sample('flat-hand'));
    $peace.addEventListener('click', () => sample('peace'));
    $thumbsUp.addEventListener('click', () => sample('thumbs-up'));
    $train.addEventListener('click', () => train());
    $save.addEventListener('click', () => classifier.save());
    $homeStart.addEventListener('click', () => {
        classifier.load(pretrainedModelURL, () => {
            classifier.normalizeData();
            setState(STATE_INFO);
        });
    });

    $create.addEventListener('click', () => {
        flatHandSamples = 0;
        thumbsUpSamples = 0;
        peaceSamples = 0;
        setState(STATE_SAMPLING);
    });

    $sampelingHome.addEventListener('click', () => {
        setState(STATE_INTRO);
    });

    $infoBack.addEventListener('click', () => {
        setState(STATE_INTRO);
    });
    $start.addEventListener('click', () => {
        setState(STATE_GAME);
    });

    $gameOverHome.addEventListener('click', () => {
        setState(STATE_INTRO);
        gameReset()
    });

    $playAgain.addEventListener('click', () => {
        setState(STATE_GAME);
        gameReset()
    });

    setState(STATE_INTRO);
}

const sample = (label) => {
    if (hands.length === 0) {
        return;
    }
    // the keypoints should be one big array of numbers
    const keypoints = getRelevantPositionKeyPoints(hands[0]).map(keypoint => [keypoint.x, keypoint.y]).flat();
    classifier.addData(keypoints, [label]);

    if (label == "peace") {
        peaceSamples++;
    } else if (label === "thumbs-up") {
        thumbsUpSamples++;
    } else if (label === "flat-hand") {
        flatHandSamples++;
    }
};

const train = () => {
    const errorTag = document.querySelector(`.error-message`);
    if (peaceSamples > 0 && flatHandSamples > 0 && thumbsUpSamples > 0) {
        errorTag.textContent = "";
        setState(STATE_TRAINING);
        

        setTimeout(() => {
            classifier.normalizeData();
            try {

                const tfModel = classifier?.model || classifier?.net || null;

                if (tfModel && typeof tfModel.compile === 'function') {

                    if (!tfModel.optimizer) {
                        tfModel.compile({
                            optimizer: 'adam',
                            loss: 'categoricalCrossentropy',
                            metrics: ['accuracy'],
                        });
                        console.info('Compiled tf model for training.');
                    }
                }
            } catch (e) {
                console.warn('Could not auto-compile tf model:', e);
            }

            const options = { epochs: 150 };
            classifier.train(options, whileTraining, finishedTraining);
        }, 50);
    } else {
        if (peaceSamples === 0 && flatHandSamples === 0 && thumbsUpSamples === 0) {
            errorTag.textContent = "Every categorie needs at least 1 sample";
        } else {
            if (peaceSamples === 0 && flatHandSamples === 0) {
                errorTag.textContent = "Peace and Flat Hand still need at least 1 sample";
            } else if (thumbsUpSamples === 0 && flatHandSamples === 0) {
                errorTag.textContent = "Thumbs Up and Flat Hand still need at least 1 sample";
            } else if (peaceSamples === 0 && thumbsUpSamples === 0) {
                errorTag.textContent = "Peace and Thumbs Up still need at least 1 sample";
            } else {
                if (peaceSamples === 0) {
                    errorTag.textContent = "Peace still need at least 1 sample";
                } else if (flatHandSamples === 0) {
                    errorTag.textContent = "Flat Hand still need at least 1 sample";
                } else if (thumbsUpSamples === 0) {
                    errorTag.textContent = "Thumbs Up still need at least 1 sample";
                }
            }
        }
    }
};

const whileTraining = (epoch, loss) => {
    $training.textContent = `Epoch: ${epoch}, Loss: ${loss.loss}`;
};

const finishedTraining = () => {
    setState(STATE_INFO);
};

const draw = () => {
    if (state == STATE_LOADING) {
        drawLoading();
    } else if (state === STATE_SAMPLING) {
        drawSampling();
    } else if (state === STATE_TRAINING) {
        drawTraining();
    } else if (state === STATE_GAME) {
        drawGame();
    }
    requestAnimationFrame(draw);
}

const drawLoading = () => {
    const $loadingLoad = document.querySelector(`#loading-load`);
    $loadingLoad.play();
};

const drawSampling = () => {
    drawVideoWithKeyPoints();
};

const drawTraining = () => {
    const $trainLoad = document.querySelector(`#train-load`);
    $trainLoad.play();
    drawVideoWithKeyPoints();
};

const progressBarDraw = () => {
    rightPattern.forEach(sign => {
        document.querySelector(`.progress-bar`).removeChild(document.querySelector(`.progress-bar`).lastElementChild)
    });

    rightPattern.forEach(sign => {
        const $progressCircle = document.createElement(`img`);
        $progressCircle.setAttribute(`src`, circleEmpty)
        document.querySelector(`.progress-bar`).appendChild($progressCircle);
    });
}

const addSign = () => {
    rightPattern.forEach(sign => {
        document.querySelector(`.progress-bar`).removeChild(document.querySelector(`.progress-bar`).lastElementChild)
    });

    rightPattern.push(signs[Math.floor(Math.random() * 3)]); //adds a new sign
    signPattern = []; //clearing the shown signs from previous zound

    document.querySelector(`.round-counter`).textContent = "Round " + rightPattern.length;
    round++;

    signOfRound = 0;

    rightPattern.forEach(sign => {
        const $progressCircle = document.createElement(`img`);
        $progressCircle.setAttribute(`src`, circleEmpty)
        document.querySelector(`.progress-bar`).appendChild($progressCircle);
    });

    //preps everything to go to the memorizeStage
    gameStage = "memorizeStage";
    memorizeSign = -1;
    timeToShowMemorize = performance.now() + 2000;
    document.querySelector(`.cam__image`).setAttribute(`src`, getReadyImg);
    document.querySelector(`.game-bottom`).style.opacity = 0;
};

const showNextSign = () => {
    const $imgTag = document.querySelector(`.cam__image`);
    if (memorizeSign === -1) {
        $imgTag.setAttribute(`src`, getReadyImg);
    } else if (rightPattern[memorizeSign] === "flat-hand") {
        $imgTag.setAttribute(`src`, flatHandMemorizeImg);
    } else if (rightPattern[memorizeSign] === "thumbs-up") {
        $imgTag.setAttribute(`src`, thumbsUpMemorizeImg);
    } else if (rightPattern[memorizeSign] === "peace") {
        $imgTag.setAttribute(`src`, peaceMemorizeImg);
    }

    progressBarDraw();

    //looks for the what sign is getting showed to higlight it in the progress bar
    if (memorizeSign > -1 && !(memorizeSign + 1 > rightPattern.length)) {
        document.querySelector(`.progress-bar img:nth-child(${memorizeSign + 1})`).setAttribute(`src`, circleFilled)
    }

    memorizeSign++;
};

const memorizeStage = () => {
    document.querySelector(`.cam__image`).style.opacity = 1;

    if (memorizeSign === rightPattern.length + 1) {
        //preps everything to go to the guessStage
        gameStage = "guessStage";
        timeToCheck = performance.now() + 3000;
        document.querySelector(`.cam__image`).style.opacity = 0;
        document.querySelector(`.game-bottom`).style.opacity = 1;
    }

    const now = performance.now();
    const diff = timeToShowMemorize - now;
    if (diff <= 0) {
        if (!(memorizeSign === rightPattern.length) && !(memorizeSign === -1)) {
            document.querySelector(`.cam__image`).setAttribute(`src`, blackMemorizeImg);
        }
        setTimeout(showNextSign, 100)
        timeToShowMemorize = now + 2000;
    }
};

const setImageAndLabel = (checkedItem, nameTag, imageClass, path) => {
    const imgMap = {
        "thumbs-up": {
            label: "Thumbs Up",
            "-Game": thumbsUpGameImg,
            "-Game-Over": thumbsUpGameOverImg,
            default: thumbsUpMemorizeImg,
        },
        "flat-hand": {
            label: "Flat Hand",
            "-Game": handGameImg,
            "-Game-Over": handGameOverImg,
            default: flatHandMemorizeImg,
        },
        "peace": {
            label: "Peace",
            "-Game": peaceGameImg,
            "-Game-Over": peaceGameOverImg,
            default: peaceMemorizeImg,
        },
    };

    const item = imgMap[checkedItem];
    if (!item) return;

    nameTag.textContent = item.label;
    document.querySelector(imageClass).setAttribute("src", item[path] || item.default);
};

const drawGameOver = (wrongAnswer) => {
    setImageAndLabel(wrongAnswer, document.querySelector(`.wrong-sign__name`), ".wrong-sign__image", "-Game-Over");

    if (!(document.querySelector(`.game-over__bar`).firstChild === null)) {
        document.querySelector(`.game-over__bar`).textContent = "";
    }

    for (let i = 0; i < rightPattern.length; i++) {
        const $progressCircle = document.createElement(`img`);
        if (i === signOfRound) {
            $progressCircle.setAttribute(`src`, circleRed);
        } else if (i < signOfRound) {
            $progressCircle.setAttribute(`src`, circleGreen);
        } else if (i > signOfRound) {
            $progressCircle.setAttribute(`src`, circleEmpty);
        }
        $progressCircle.setAttribute(`alt`, i);
        document.querySelector(`.game-over__bar`).appendChild($progressCircle);
    }

    if (round - 1 === 0) {
        round = "00";
    } else if (!(round - 1 > 9)) {
        round = `0${round - 1}`;
    }

    document.querySelector(`.game-over__digit`).textContent = round;

    setImageAndLabel(rightPattern[signOfRound], document.querySelector(`.right-sign__name`), ".right-sign__image", "-Game-Over");
}

const check = () => {
    if (classificationResults[0]?.label === rightPattern[signOfRound]) {
        const correctAudio = new Audio(correctSound);
        correctAudio.play();

        signPattern.push(classificationResults[0]?.label);
        signOfRound++;

        document.querySelector(`.progress-bar`).removeChild(document.querySelector(`.progress-bar`).lastElementChild)
        const $progressCircle = document.createElement(`img`);
        $progressCircle.setAttribute(`src`, circleGreen)
        document.querySelector(`.progress-bar`).insertBefore($progressCircle, document.querySelector(`.progress-bar`).firstChild);
    } else {
        const wrongAudio = new Audio(wrongSound);
        wrongAudio.play();

        setState(STATE_GAME_OVER);
        drawGameOver(classificationResults[0]?.label)
    }
};

const guessStage = () => {
    //highlights the sign that needs to be guessed in the progress bar
    document.querySelector(`.progress-bar img:nth-child(${signOfRound + 1})`).setAttribute(`src`, circleFilled)

    const now = performance.now();
    const diff = timeToCheck - now;
    document.querySelector(`.count-down`).textContent = Math.ceil(diff / 1000);
    if (diff <= 0) {
        check();
        timeToCheck = now + 3000;
    }
};

const drawGame = () => {
    drawVideoWithKeyPoints();
    setImageAndLabel(classificationResults[0]?.label, $results, ".answer-img", "-Game");

    if (rightPattern.length === signPattern.length) {
        addSign();
    }

    if (gameStage === "memorizeStage") {
        memorizeStage()
    } else {
        guessStage()
    }

};

const drawVideoWithKeyPoints = () => {
    ctx.drawImage(video, 0, 0, $canvas.width, $canvas.height);
};

const getRelevantPositionKeyPoints = (hand) => {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;

    hand.keypoints.forEach(point => {
        if (point.x < minX) {
            minX = point.x
        }
        if (point.y < minY) {
            minY = point.y
        }
    });

    const middleFingerMcp = {
        x: hand.middle_finger_mcp.x,
        y: hand.middle_finger_mcp.y,
    };
    const handWrist = {
        x: hand.wrist.x,
        y: hand.wrist.y,
    };
    // calculate the distance between the eyes, we use this as a distance-based scaling factor
    const handDistance = Math.sqrt(
        Math.pow(middleFingerMcp.x - handWrist.x, 2) +
        Math.pow(middleFingerMcp.y - handWrist.y, 2)
    );
    const targetDistance = 100;
    const keypointMap = (keypoint) => {
        return {
            x: targetDistance * (keypoint.x - minX + 50) / handDistance,
            y: targetDistance * (keypoint.y - minY + 50) / handDistance,
            z: keypoint.z
        }
    };
    return [
        //... = spread operator = They 'unpack' the items from an array or the properties of an object and 'spread' them as individual elements
        ...hand.keypoints.map(keypointMap),
    ];
};

preload();