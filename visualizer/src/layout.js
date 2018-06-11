import kiwi from 'kiwi.js'


// Shorthands for Kiwi.js.
const kvar = kiwi.Variable
const kexp = kiwi.Expression
const keq = kiwi.Operator.Eq
const kge = kiwi.Operator.Ge
const kle = kiwi.Operator.Le
const kstrong = kiwi.Strength.strong
const kweak = kiwi.Strength.weak


// Positionning constants, in pixels.
const leftPadding = 0
const rightPadding = 0
const topPadding = 90
const bottomPadding = 0
const xNodeSep = 90
const yNodeSep = 60
const xConstSep = 80
const yConstSep = 30


/**
 * A custom Cytoscape layout for Tensorflow graphs.
 *
 * It uses the Kiwi.js linear solver to express and optimize
 * the linear constraints for the spacing between nodes, and
 * falls back to the Dagre layout algorithm if the linear
 * solver fails to find an optimal solution.
 */
class TensorflowLayout {
  constructor(options) {
    this.options = options
    this.cy = options.cy
    this.eles = options.eles
    this.nodes = options.eles.nodes()
    this.edges = options.eles.edges()

    let bb = options.boundingBox || {
      x1: 0,
      y1: 0,
      w: this.cy.width(),
      h: this.cy.height()
    }

    if (bb.x2 === undefined){ bb.x2 = bb.x1 + bb.w; }
    if (bb.w === undefined){ bb.w = bb.x2 - bb.x1; }
    if (bb.y2 === undefined){ bb.y2 = bb.y1 + bb.h; }
    if (bb.h === undefined){ bb.h = bb.y2 - bb.y1; }

    this.bb = bb
  }

  run() {
    let idx = {}
    this.nodes.forEach((n, i) => idx[n.id()] = i)

    let positions = {}

    try {
      positions = this.solveLinear(idx)
    } catch (e) {
      positions = this.solveDagre(idx)
    }

    this.nodes.layoutPositions(this, this.options, function(node){
      node = typeof node === "object" ? node : this
      return positions[idx[node.id()]]
    })

    return this
  }

  /** Tries to position the nodes using the linear solver. */
  solveLinear(idx) {
    console.log('Solving using the linear solver.')

    let nodes = this.nodes
    let solver = new kiwi.Solver()
    let xs = nodes.map(_ => new kvar())
    let ys = nodes.map(_ => new kvar())

    nodes.forEach(n => {
      let nid = idx[n.id()]

      solver.addEditVariable(xs[nid], kstrong)
      solver.addEditVariable(ys[nid], kstrong)
      solver.suggestValue(xs[nid], this.bb.w / 2)
    })

    // Ensure the nodes stay in the bounding box.
    nodes.forEach(n => {
      let nid = idx[n.id()]

      solver.createConstraint(xs[nid], kge, this.bb.x1 + leftPadding)
      solver.createConstraint(xs[nid], kle, this.bb.x2 + rightPadding, kstrong)

      solver.createConstraint(ys[nid], kge, this.bb.y1 + topPadding)
      solver.createConstraint(ys[nid], kle, this.bb.y2 + bottomPadding, kstrong)
    })

    // Add constraints to predecessors.
    nodes.forEach(n => {
      let nid = idx[n.id()]

      /**
       * Rules for non-constant predecessors.
       */
      let prevNodes = n.incomers().filter('node[op != "Const"]')
      let prevNodesSize = prevNodes.length

      prevNodes.forEach((m, i) => {
        let mid = idx[m.id()]

        // The successor is below all the predecessors.
        solver.createConstraint(ys[nid], kge, new kexp(ys[mid], yNodeSep))

        // All the predecessors are spaced from each other.
        if (i > 0) {
          let pid = idx[prevNodes[i - 1].id()]
          solver.createConstraint(xs[mid], kge, new kexp(xs[pid], xNodeSep))
        }
      })

      // The successor is aligned with the middle of the predecessors.
      if (prevNodesSize > 0) {
        let xmean = prevNodes.map(m => [1 / prevNodesSize, xs[idx[m.id()]]])
        solver.createConstraint(xs[nid], keq, new kexp(...xmean), kstrong)
      }

      /**
       * Rules for constant predecessors.
       */
      let prevConsts = n.incomers().filter('node[op = "Const"]')
      let prevConstsSize = prevConsts.length

      prevConsts.forEach((m, i) => {
        let mid = idx[m.id()]

        // The successor is left of all the constant predecessors.
        solver.createConstraint(xs[nid], kge, new kexp(xs[mid], xConstSep))

        // All the constant predecessors are spaced from each other.
        if (i > 0) {
          let pid = idx[prevConsts[i - 1].id()]
          solver.createConstraint(ys[mid], kge, new kexp(ys[pid], yConstSep))
        }
      })

      // The successor is aligned with the middle of the predecessors.
      if (prevConstsSize > 0) {
        let ymean = prevConsts.map(m => [1 / prevConstsSize, ys[idx[m.id()]]])
        solver.createConstraint(ys[nid], keq, new kexp(...ymean))
      }
    })

    solver.updateVariables()

    return nodes.map(node => ({
      x: xs[idx[node.id()]].value(),
      y: ys[idx[node.id()]].value(),
    }))
  }

  /** Tries to position the nodes using Dagre. */
  solveDagre(idx) {
    console.log('Solving using Dagre.')
    // TODO(liautaud)
  }
}

// Export the layout as a Cytoscape extension.
export default (cytoscape) => {
  cytoscape('layout', 'tensorflow', TensorflowLayout)
}