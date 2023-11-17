function SelectImage(name) {
    let input = document.getElementById(`${name}-file`)
    input.click()
}

function UpdateLabel(name, percent = false) {
    let value = +document.getElementById(name).value
    let span = document.getElementById(`${name}-value`)

    if (percent)
        span.innerText = `${Math.round(value * 10000) / 100}%`
    else
        span.innerText = `${value}`
}

function UpdateIgnoreTarget() {
    let method = document.getElementById("method").value
    let ignoreTarget = document.getElementById("ignore-target").checked
    let paramsBlock = document.getElementById("qp-params-block")

    if (ignoreTarget || method != "qp")
        paramsBlock.classList.add("hidden")
    else
        paramsBlock.classList.remove("hidden")
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

function ShowVector(vector) {
    vector = vector.map(v => `${Math.round(v * 10000) / 10000}`)
    return vector.join(", ")
}

function Softmax(vector) {
    let sum = 0
    let result = []

    for (let i = 0; i < vector.length; i++) {
        result.push(Math.exp(vector[i]))
        sum += result[i]
    }

    for (let i = 0; i < vector.length; i++)
        result[i] /= sum

    return result
}

function ShowPrediction(image, block, prediction, name) {
    let imageFullSize = document.getElementById(`${name}-image`)
    imageFullSize.src  = `data:image/jpeg;base64,${image}`

    let imageRealSize = document.getElementById(`${name}-real-image`)
    imageRealSize.src  = `data:image/jpeg;base64,${image}`

    block.parentNode.classList.remove("hidden")
    block.innerHTML = `
        <div class="text"><b>Диапазон входа:</b> ${Math.round(prediction.x_min * 10000) / 10000}...${Math.round(prediction.x_max * 10000) / 10000}</div>
        <div class="text"><b>Выход сети:</b> ${ShowVector(prediction.output)}</div>
        <div class="text"><b>Выход первого слоя:</b> ${ShowVector(prediction.first_layer)}</div>
    `

    let svg = document.getElementById(`${name}-barchart`)
    svg.parentNode.classList.remove("hidden")

    let barChart = new BarChart(svg)
    barChart.Plot(Softmax(prediction.output), prediction.class_names)
}

function ChangeMethod() {
    let method = document.getElementById("method").value
    let paramsBlock = document.getElementById("qp-params-block")

    if (method == "qp")
        paramsBlock.classList.remove("hidden")
    else
        paramsBlock.classList.add("hidden")
}

function Predict(name) {
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

        ShowPrediction(response.image, prediction, response.prediction, name)
        ShowAttackButton()
    })
}

function Attack() {
    let image = document.getElementById("attack-image")
    let imageRealSize = document.getElementById("attack-real-image")
    let prediction = document.getElementById("attack-prediction")
    let method = document.getElementById("method").value
    let ignoreTarget = document.getElementById("ignore-target").checked
    let scale = +document.getElementById("scale").value
    let mask = document.getElementById("mask").value
    let pixelDiff = +document.getElementById("pixel-diff").value
    let error = document.getElementById("error")
    error.innerText = ""

    let data = new FormData()
    data.append("input_image", document.getElementById("input-file").files[0])
    data.append("target_image", document.getElementById("target-file").files[0])
    data.append("scale", scale)
    data.append("ignore_target", ignoreTarget)
    data.append("mask", mask)
    data.append("pixel_diff", pixelDiff)
    data.append("method", method)

    SendRequest("/attack", data).then(response => {
        if (response.status != "success") {
            error.innerText = response.message
            prediction.parentNode.classList.add("hidden")
            return
        }

        ShowPrediction(response.image, prediction, response.prediction, "attack")
    })
}
