/** Returns the path of the predecessor. */
function predecessor(path) {
  return path.split('/').slice(0, -1).join('/')
}

export class Hierarchy {
  constructor(nodes) {
    // The node path corresponding to each node index.
    this.indexMapping = {}

    // Whether a node with a given path exists.
    this.existsMapping = {}

    // The metanode path corresponding to each node path.
    this.metanodeMapping = {}

    nodes.forEach(n => {
      let path = n.name
      let pred = predecessor(path)

      this.indexMapping[n.id] = path
      this.existsMapping[path] = true

      if (pred) {
        this.metanodeMapping[pred] = false
      }
    })

    Object.keys(this.existsMapping).forEach(k => {
      delete this.metanodeMapping[k]
    })

    Object.keys(this.metanodeMapping).forEach(k => {
      // Group the metanodes which share the same prefix.
      let p = k.split('/')
      let pp = p.pop()

      let q = pp.split('_')
      let qq = q.pop()
      if (q.length == 0) {
        return
      }

      let base = p.join('/') + '/' + q.join('_')
      if (base in this.metanodeMapping) {
        this.metanodeMapping[k] = base
      }
    })
  }

  getName(path) {
    return path.split('/').pop()
  }

  getParent(path) {
    let parent = predecessor(path)

    if (parent.length == 0) {
      return null
    }

    // A parent should be a metanode, not a regular one.
    if (parent in this.metanodeMapping) {
      if (this.metanodeMapping[parent]) {
        return this.metanodeMapping[parent]
      } else {
        return parent
      }
    }

    // Otherwise, we reach for the grandparent.
    return this.getParent(parent)
  }

  getPath(index) {
    return this.indexMapping[index]
  }

  getMetanodes() {
    return Object.keys(this.metanodeMapping)
    .filter(k => !this.metanodeMapping[k])
    .map(k => {
      return {
        data: {
          id: k,
          name: k,
          oid: -1,
          parent: this.getParent(k),
        }
      }
    })
  }
}
