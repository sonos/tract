import kiwi from 'kiwi.js'


// Shorthands for Kiwi.js.
const kvar = kiwi.Variable
const kexp = kiwi.Expression
const keq = kiwi.Operator.Eq
const kge = kiwi.Operator.Ge
const kle = kiwi.Operator.Le
const kstrong = kiwi.Strength.strong


// Positionning constants, in pixels.
const xConstSep = 55
const yConstSep = 40


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

  /** Position the nodes using a combination of the Dagre solver and a linear one. */
  run() {
    let nodes = this.nodes.filter('node[op != "Const"]')
    let eles = nodes.union(nodes.edgesTo(nodes))

    // Position the non-constant nodes using Dagre.
    let dagre = eles.layout({
      name: 'dagre',
      fit: false,
      // TODO(liautaud)
    })

    dagre.run()

    // Position the constant nodes using the linear solver.
    let consts = this.nodes.filter('node[op = "Const"]')
    let solver = new kiwi.Solver()
    let xs = consts.map(_ => new kvar())
    let ys = consts.map(_ => new kvar())

    let idx = {}
    consts.forEach((n, i) => idx[n.id()] = i)

    consts.forEach(n => {
      let nid = idx[n.id()]

      solver.addEditVariable(xs[nid], kstrong)
      solver.addEditVariable(ys[nid], kstrong)
    })

    consts.outgoers('node').forEach(n => {
      let nx = n.position('x')
      let ny = n.position('y')
      let nw = n.layoutDimensions().w
      let nh = n.layoutDimensions().h

      let prevConsts = n.incomers().filter('node[op = "Const"]')
      let prevConstsSize = prevConsts.length

      prevConsts.forEach((m, i) => {
        let mid = idx[m.id()]

        let leftx = nx - (xConstSep + nw / 2);
        let rightx = nx + (xConstSep + nw / 2);
        let conflicts = nodes.some(o => {
          // Ignore expanded meta-nodes.
          if (o.isParent() && !o.hasClass('cy-expand-collapse-collapsed-node')) {
            return false;
          }

          let ox = o.position('x')
          let oy = o.position('y')
          let ow = o.layoutDimensions().w
          let oh = o.layoutDimensions().h

          return ox - ow / 2 <= leftx &&
                 leftx <= ox + ow / 2 &&
                 oy - oh / 2 <= ny &&
                 ny <= oy + oh / 2
        })

        // The constants are left of the successor if there is no conflict,
        // and to its right otherwise.
        if (!conflicts) {
          solver.createConstraint(xs[mid], keq, leftx)
        } else {
          solver.createConstraint(xs[mid], keq, rightx)
        }

        // All the constants are spaced from each other.
        if (i > 0) {
          let pid = idx[prevConsts[i - 1].id()]
          solver.createConstraint(ys[mid], kge, new kexp(ys[pid], yConstSep))
        }
      })

      // The middle of the constants is aligned with the successor.
      let ymean = prevConsts.map(m => [1 / prevConstsSize, ys[idx[m.id()]]])
      solver.createConstraint(new kexp(...ymean), keq, ny)
    })

    solver.updateVariables()
    consts.layoutPositions(this, this.options, function(node){
      node = typeof node === "object" ? node : this
      return {
        x: xs[idx[node.id()]].value(),
        y: ys[idx[node.id()]].value(),
      }
    })

    return this
  }
}

// Export the layout as a Cytoscape extension.
export default (cytoscape) => {
  cytoscape('layout', 'tensorflow', TensorflowLayout)
}