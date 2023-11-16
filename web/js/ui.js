function SelectImage(name) {
    let input = document.getElementById(`${name}-file`)
    input.value = null
    input.click()
}

function ShowAttackButton() {
    let haveInput = document.getElementById("input-file").files.length == 1
    let haveTarget = document.getElementById("target-file").files.length == 1
    let controls = document.getElementById("controls")

    if (haveInput && haveTarget)
        controls.classList.remove("hidden")
    else
        controls.classList.add("hidden")
}

function RoundVector(vector) {
    return vector.map(v => Math.round(v * 10000) / 10000)
}

function ShowPrediction(block, prediction) {
    let html = [
        `<b>Выход сети:</b> ${RoundVector(prediction.output).join(", ")}`,
        `<b>Выход первого слоя:</b> ${RoundVector(prediction.first_layer).join(", ")}`
    ]
    block.innerHTML = html.join("<br>")
}

function Predict(name) {
    let image = document.getElementById(`${name}-image`)
    let input = document.getElementById(`${name}-file`)
    let prediction = document.getElementById(`${name}-prediction`)
    let error = document.getElementById(`${name}-error`)
    error.innerText = ""

    let data = new FormData()
    data.append("image", input.files[0])

    SendRequest("/predict", data).then(response => {
        if (response.status != "success") {
            error.innerText = response.message
            return
        }

        image.src  = `data:image/jpeg;base64,${response.image}`
        ShowPrediction(prediction, response.prediction)
        ShowAttackButton()
    })
}

function Attack() {
    let image = document.getElementById("attack-image")
    let prediction = document.getElementById("attack-prediction")
    let error = document.getElementById("error")
    error.innerText = ""

    let data = new FormData()
    data.append("input_image", document.getElementById("input-file").files[0])
    data.append("target_image", document.getElementById("target-file").files[0])

    SendRequest("/attack", data).then(response => {
        if (response.status != "success") {
            error.innerText = response.message
            return
        }

        image.src  = `data:image/jpeg;base64,${response.image}`
        ShowPrediction(prediction, response.prediction)
    })
}