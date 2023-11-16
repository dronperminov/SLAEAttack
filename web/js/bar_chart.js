function BarChart(svg) {
    this.svg = svg
    this.maxRectWidth = 50
    this.padding = {top: 10, bottom: 70, horizontal: 5}
}

BarChart.prototype.MakeBar = function(x, y, rectWidth, rectHeight) {
    let bar = document.createElementNS('http://www.w3.org/2000/svg', "rect")
    bar.setAttribute("x", x)
    bar.setAttribute("y", y)
    bar.setAttribute("width", rectWidth)
    bar.setAttribute("height", rectHeight)
    bar.setAttribute("class", "bar")
    return bar
}

BarChart.prototype.AppendLabel = function(x, y, labelText, baseline = "middle") {
    let label = document.createElementNS('http://www.w3.org/2000/svg', "text")
    label.textContent = labelText
    label.setAttribute("x", x)
    label.setAttribute("y", y)
    label.setAttribute("alignment-baseline", baseline)
    label.setAttribute("text-anchor", "start")
    label.setAttribute("writing-mode", "tb")
    label.setAttribute("letter-spacing", "-0.5")
    this.svg.appendChild(label)
}

BarChart.prototype.AppendBar = function(x, y, rectWidth, rectHeight) {
    this.svg.appendChild(this.MakeBar(x, y, rectWidth, rectHeight))
}

BarChart.prototype.Plot = function(values, classNames, min = null, max = null) {
    this.svg.innerHTML = ''

    let width = this.svg.clientWidth
    let height = this.svg.clientHeight
    let rectWidth = width / values.length - this.padding.horizontal

    if (rectWidth > this.maxRectWidth)
        rectWidth = this.maxRectWidth

    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`)

    if (min === null)
        min = Math.min(...values)

    if (max === null)
        max = Math.max(...values)

    for (let i = 0; i < values.length; i++) {
        let rectHeight = (values[i] - min) / (max - min) * (height - this.padding.top - this.padding.bottom)
        let x = this.padding.horizontal / 2 + i * (this.padding.horizontal + rectWidth)
        let y = height - this.padding.bottom - rectHeight

        this.AppendBar(x, y, rectWidth, rectHeight)
        this.AppendLabel(x + rectWidth / 2, height - this.padding.bottom, `${classNames[i]}`)
    }
}
