package matrix

import "fmt"

func Multiply(a [][]float64, b [][]float64) ([][]float64, error) {
	if len(a) == 0 || len(a[0]) == 0 || len(b) == 0 || len(b[0]) == 0 {
		return nil, fmt.Errorf("invalid array input")
	}
	if len(a[0]) != len(b) {
		return nil, fmt.Errorf("row length of first elem must be same as column length of second elem")
	}
	newMatrix := make([][]float64, len(a))
	for yIndex := range newMatrix {
		newMatrix[yIndex] = make([]float64, len(b[0]))
		for xIndex := range newMatrix[yIndex] {
			for iter := range b {
				newMatrix[yIndex][xIndex] += a[yIndex][iter] * b[iter][xIndex]
			}
		}
	}
	return newMatrix, nil
}
