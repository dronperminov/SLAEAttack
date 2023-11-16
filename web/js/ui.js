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

function ShowVector(vector, withMax = false) {
    vector = vector.map(v => `${Math.round(v * 10000) / 10000}`)

    if (withMax) {
        let imax = 0
        let imin = 0

        for (let i = 1; i < vector.length; i++) {
            if (vector[i] > vector[imax])
                imax = i

            if (vector[i] < vector[imin])
                imin = i
        }

        vector[imin] = `<span class="argmin">${vector[imin]}</span>`
        vector[imax] = `<span class="argmax">${vector[imax]}</span>`
    }

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

    block.innerHTML = `
        <div class="text"><b>Выход сети:</b> ${ShowVector(prediction.output, true)}</div>
        <div class="text"><b>Выход первого слоя:</b> ${ShowVector(prediction.first_layer)}</div>
    `

    let svg = document.getElementById(`${name}-barchart`)
    svg.parentNode.classList.remove("hidden")

    let barChart = new BarChart(svg)
    barChart.Plot(Softmax(prediction.output))
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

        ShowPrediction(response.image, prediction, response.prediction, "attack")
    })
}
