import { Model } from './model.js'
import { loadData } from './data.js'

const [X, y] = loadData()
const m = new Model()
m.fit(X, y)
console.log(m.predict(X.slice(0,5)))
