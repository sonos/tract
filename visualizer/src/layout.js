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
const xNodeSep = 150
const yNodeSep = 40
const xConstSep = 40
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

    this.dagre = this.cy.layout({
      name: 'dagre',
    })

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
    try {
      this.solveLinear()
    } catch (e) {
      this.solveDagre()
    }

    return this
  }

  /** Tries to position the nodes using the linear solver. */
  solveLinear() {
    console.log('Solving using the linear solver.')

    let nodes = this.nodes
    let solver = new kiwi.Solver()
    let xs = nodes.map(_ => new kvar())
    let ys = nodes.map(_ => new kvar())

    let idx = {}
    nodes.forEach((n, i) => idx[n.id()] = i)

    nodes.forEach(n => {
      let nid = idx[n.id()]

      solver.addEditVariable(xs[nid], kstrong)
      solver.addEditVariable(ys[nid], kstrong)
    })

    // Add constraints to predecessors.
    nodes.forEach(n => {
      let nid = idx[n.id()]
      let nw = n.layoutDimensions().w
      let nh = n.layoutDimensions().h

      /**
       * Rules for non-constant predecessors.
       */
      let prevNodes = n.incomers().filter('node[op != "Const"]')
      let prevNodesSize = prevNodes.length

      prevNodes.forEach((m, i) => {
        let mid = idx[m.id()]

        // The successor is below all the predecessors.
        solver.createConstraint(ys[nid], kge, new kexp(ys[mid], yNodeSep + nh / 2))

        // All the predecessors are spaced from each other.
        if (i > 0) {
          let p = prevNodes[i - 1]
          let pid = idx[p.id()]
          let pw = n.layoutDimensions().w
          solver.createConstraint(xs[mid], kge, new kexp(xs[pid], xNodeSep + pw / 2))
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
        solver.createConstraint(xs[nid], kge, new kexp(xs[mid], xConstSep + nw / 2))

        // All the constant predecessors are spaced from each other.
        if (i > 0) {
          let pid = idx[prevConsts[i - 1].id()]
          solver.createConstraint(ys[mid], kge, new kexp(ys[pid], yConstSep))
        }

        // All the constant predecessors are below the non-constant ones.
        prevNodes.forEach(o => {
          solver.createConstraint(ys[mid], kge, new kexp(ys[idx[o.id()]], yNodeSep / 3))
        })
      })

      // The successor is aligned with the middle of the predecessors.
      if (prevConstsSize > 0) {
        let ymean = prevConsts.map(m => [1 / prevConstsSize, ys[idx[m.id()]]])
        solver.createConstraint(ys[nid], keq, new kexp(...ymean))
      }
    })

    solver.updateVariables()

    this.nodes.layoutPositions(this, this.options, function(node){
      node = typeof node === "object" ? node : this
      return {
        x: xs[idx[node.id()]].value(),
        y: ys[idx[node.id()]].value(),
      }
    })
  }

  /** Tries to position the nodes using Dagre. */
  solveDagre() {
    console.log('Solving using Dagre.')

    // TODO(liautaud): Fix this when using collapse.
    this.dagre.run()
  }
}

// Export the layout as a Cytoscape extension.
export default (cytoscape) => {
  cytoscape('layout', 'tensorflow', TensorflowLayout)
}