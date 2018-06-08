import kiwi from 'kiwi.js'

const kvar = kiwi.Variable
const kexp = kiwi.Expression
const keq = kiwi.Operator.Eq
const kge = kiwi.Operator.Ge
const kle = kiwi.Operator.Le
const kstrong = kiwi.Strength.strong
const kweak = kiwi.Strength.weak

/**
 * A custom Cytoscape layout for Tensorflow graphs.
 *
 * It uses the Kiwi.js linear solver to express and optimize
 * the linear constraints for the spacing between nodes.
 */
class TensorflowLayout {
  constructor(options) {
    this.options = options
  }

  run() {
    let options = this.options
    let cy = options.cy
    let eles = options.eles
    let nodes = eles.nodes()
    let edges = eles.edges()

    let bb = options.boundingBox || { x1: 0, y1: 0, w: cy.width(), h: cy.height() }
    if (bb.x2 === undefined){ bb.x2 = bb.x1 + bb.w; }
    if (bb.w === undefined){ bb.w = bb.x2 - bb.x1; }
    if (bb.y2 === undefined){ bb.y2 = bb.y1 + bb.h; }
    if (bb.h === undefined){ bb.h = bb.y2 - bb.y1; }

    // Positionning constants, in pixels.
    const leftPadding = 0
    const rightPadding = 0
    const topPadding = 90
    const bottomPadding = 0
    const xNodeSep = 100
    const yNodeSep = 50
    const xConstSep = 80
    const yConstSep = 50

    // Initialize the variables of the linear problem.
    let solver = new kiwi.Solver()
    let xs = nodes.map(_ => new kvar())
    let ys = nodes.map(_ => new kvar())
    nodes.forEach(n => {
      solver.addEditVariable(xs[n.data('oid')], kstrong)
      solver.addEditVariable(ys[n.data('oid')], kstrong)
      solver.suggestValue(xs[n.data('oid')], bb.w / 2)
    })

    console.log(bb.w / 2)

    // Ensure the nodes stay in the bounding box.
    nodes.forEach(n => {
      solver.createConstraint(xs[n.data('oid')], kge, bb.x1 + leftPadding)
      solver.createConstraint(xs[n.data('oid')], kle, bb.x2 + rightPadding, kstrong)

      solver.createConstraint(ys[n.data('oid')], kge, bb.y1 + topPadding)
      solver.createConstraint(ys[n.data('oid')], kle, bb.y2 + bottomPadding, kstrong)
    })

    // Add constraints to predecessors.
    nodes.forEach(n => {
      let nid = n.data('oid')

      /**
       * Rules for non-constant predecessors.
       */
      let prevNodes = n.incomers().filter('node[op != "Const"]')
      let prevNodesSize = prevNodes.length

      prevNodes.forEach((m, i) => {
        let mid = m.data('oid')

        // The successor is below all the predecessors.
        solver.createConstraint(ys[nid], kge, new kexp(ys[mid], yNodeSep))

        // All the predecessors are spaced from each other.
        if (i > 0) {
          let pid = prevNodes[i - 1].data(oid)
          solver.createConstraint(xs[mid], kge, new kexp(xs[pid], xNodeSep))
        }
      })

      // The successor is aligned with the middle of the predecessors.
      if (prevNodesSize > 0) {
        let xmean = prevNodes.map(m => [1 / prevNodesSize, xs[m.data('oid')]])
        solver.createConstraint(xs[nid], keq, new kexp(...xmean))
      }

      /**
       * Rules for constant predecessors.
       */
      let prevConsts = n.incomers().filter('node[op = "Const"]')
      let prevConstsSize = prevConsts.length

      prevConsts.forEach((m, i) => {
        let mid = m.data('oid')

        // The successor is left of all the constant predecessors.
        solver.createConstraint(xs[nid], kge, new kexp(xs[mid], xConstSep))

        // All the constant predecessors are spaced from each other.
        if (i > 0) {
          let pid = prevConsts[i - 1].data(oid)
          solver.createConstraint(ys[mid], kge, new kexp(ys[pid], yConstSep))
        }
      })

      // The successor is aligned with the middle of the predecessors.
      if (prevConstsSize > 0) {
        let ymean = prevConsts.map(m => [1 / prevConstsSize, ys[m.data('oid')]])
        solver.createConstraint(ys[nid], keq, new kexp(...ymean))
      }
    })

    // solver.createConstraint(xs[0], kge, bb.w / 2)
    solver.suggestValue(xs[0], bb.w / 2)
    solver.suggestValue(ys[0], 0)
    solver.updateVariables()

    nodes.layoutPositions(this, options, function(node){
      node = typeof node === "object" ? node : this
      console.log(node.id(), xs[node.data('oid')].value(), ys[node.data('oid')].value())
      return {
        x: xs[node.data('oid')].value(),
        y: ys[node.data('oid')].value(),
      }
    })

    return this
  }
}

// Export the layout as a Cytoscape extension.
export default (cytoscape) => {
  cytoscape('layout', 'tensorflow', TensorflowLayout)
}