package neuralnet

import (
	"math/rand"
	"neuralNet/utils/function"
	"neuralNet/utils/matrix"
	"time"
)

type Net struct {
	inputNodes     int
	hiddenNodes    int
	outputNodes    int
	learningRate   float64
	activeFunction func(float64) float64
	wih            [][]float64
	who            [][]float64
}

func New(inputNodes, hiddenNodes, outputNodes int, learningRate float64, activeFunction func(float64) float64) *Net {
	if activeFunction == nil {
		activeFunction = function.Sigmoid
	}
	return &Net{
		inputNodes:     inputNodes,
		hiddenNodes:    hiddenNodes,
		outputNodes:    outputNodes,
		learningRate:   learningRate,
		activeFunction: activeFunction,
		wih:            newWeight(hiddenNodes, inputNodes),
		who:            newWeight(outputNodes, hiddenNodes),
	}
}

func (n *Net) Query(rawData []float64) ([]float64, error) {
	data := [][]float64{rawData}
	data = matrix.Reverse(data)

	hiddenValue, err := matrix.Multiply(n.wih, data)
	if err != nil {
		return nil, err
	}
	for yIndex := range hiddenValue {
		for xIndex := range hiddenValue[yIndex] {
			hiddenValue[yIndex][xIndex] = n.activeFunction(hiddenValue[yIndex][xIndex])
		}
	}

	finalValue, err := matrix.Multiply(n.who, hiddenValue)
	if err != nil {
		return nil, err
	}
	for yIndex := range finalValue {
		for xIndex := range finalValue[yIndex] {
			finalValue[yIndex][xIndex] = n.activeFunction(finalValue[yIndex][xIndex])
		}
	}
	result := matrix.Reverse(finalValue)
	return result[0], nil
}

func (n *Net) Train(rawData []float64, rawTargets []float64) error {
	data := [][]float64{rawData}
	data = matrix.Reverse(data)

	target := [][]float64{rawTargets}
	target = matrix.Reverse(target)

	hiddenValue, err := matrix.Multiply(n.wih, data)
	if err != nil {
		return err
	}
	for yIndex := range hiddenValue {
		for xIndex := range hiddenValue[yIndex] {
			hiddenValue[yIndex][xIndex] = n.activeFunction(hiddenValue[yIndex][xIndex])
		}
	}

	finalValue, err := matrix.Multiply(n.who, hiddenValue)
	if err != nil {
		return err
	}
	for yIndex := range finalValue {
		for xIndex := range finalValue[yIndex] {
			finalValue[yIndex][xIndex] = n.activeFunction(finalValue[yIndex][xIndex])
		}
	}

	errors := make([][]float64, len(target))
	for yIndex := range errors {
		errors[yIndex] = make([]float64, len(target[yIndex]))
		for xIndex := range errors[yIndex] {
			errors[yIndex][xIndex] = target[yIndex][xIndex] - finalValue[yIndex][xIndex]
		}
	}
	hiddenErrors, err := matrix.Multiply(matrix.Reverse(n.who), errors)
	if err != nil {
		return err
	}

	temp := make([][]float64, len(errors))
	for yIndex := range temp {
		temp[yIndex] = make([]float64, len(errors[yIndex]))
		for xIndex := range temp[yIndex] {
			temp[yIndex][xIndex] = errors[yIndex][xIndex] * finalValue[yIndex][xIndex] * (1 - finalValue[yIndex][xIndex])
		}
	}
	delta, err := matrix.Multiply(temp, matrix.Reverse(hiddenValue))
	if err != nil {
		return err
	}
	for yIndex := range n.who {
		for xIndex := range n.who[yIndex] {
			n.who[yIndex][xIndex] += delta[yIndex][xIndex]
		}
	}

	temp = make([][]float64, len(hiddenErrors))
	for yIndex := range temp {
		temp[yIndex] = make([]float64, len(hiddenErrors[yIndex]))
		for xIndex := range temp[yIndex] {
			temp[yIndex][xIndex] = hiddenErrors[yIndex][xIndex] * hiddenValue[yIndex][xIndex] * (1 - hiddenValue[yIndex][xIndex])
		}
	}
	delta, err = matrix.Multiply(temp, matrix.Reverse(data))
	if err != nil {
		return err
	}
	for yIndex := range n.wih {
		for xIndex := range n.wih[yIndex] {
			n.wih[yIndex][xIndex] += delta[yIndex][xIndex]
		}
	}
	return nil
}

func newWeight(height, width int) [][]float64 {
	w := make([][]float64, height)
	rand.Seed(time.Now().UnixNano())
	for yIndex := range w {
		w[yIndex] = make([]float64, width)
		for xIndex := range w[yIndex] {
			w[yIndex][xIndex] = newRandomWeightValue()
		}
	}
	return w
}

func newRandomWeightValue() float64 {
	var i float64
	for i = rand.NormFloat64() * 0.3; i >= 1; i = rand.NormFloat64() * 0.3 {
	}
	return i
}
