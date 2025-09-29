export class Model {
  fit(X, y) { console.log('Fitting...') }
  predict(X) { return Array(X.length).fill(0) }
}
