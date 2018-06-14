/** Returns the path of the predecessor. */
function predecessor(path) {
  return path.split('/').slice(0, -1).join('/')
}

/** Returns the hex color code corresponding to a string. */
export function stringToColor(str) {
  var hash = 0;
  for (var i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  return hash % 360
}

/** Returns the string corresponding to a type fact. */
export function typeToString(type) {
  if (!type.Only) {
    return 'Any'
  }

  return type.Only
}

/** Returns the string corresponding to a shape fact. */
export function shapeToString(shape) {
  let dims = shape.dims.map(d => (d.Only) ? d.Only.toString() : '_')

  if (shape.open) {
    dims.push('...')
  }

  return '[' + dims.join(', ') + ']'
}

export class Hierarchy {
  constructor(nodes) {
    // The node path corresponding to each node index.
    this.indexMapping = {}

    // All the paths in the graph.
    this.paths = {}

    // Add all the paths to the list recursively.
    nodes.forEach(n => {
      let parts = n.name.split('/')
      let length = parts.length
      this.indexMapping[n.id] = n.name

      while (parts.length > 0) {
        let mapping = parts.join('/')
        let isMetanode = true
        let isUsed = true

        this.paths[mapping] = { isMetanode, isUsed, mapping }
        parts.pop()
      }
    })

    // Mark all the nodes which actually exist.
    nodes.forEach(n => {
      this.paths[n.name].isMetanode = false
    })

    // Sort all the paths by depth, so that we can apply
    // transformations recursively starting with parents.
    let pathsByDepth = {}

    Object.keys(this.paths).forEach(k => {
      let depth = k.split('/').length

      if (!(depth in pathsByDepth)) {
        pathsByDepth[depth] = []
      }

      pathsByDepth[depth].push(k)
    })

    Object.keys(pathsByDepth).forEach(depth => {
      pathsByDepth[depth].forEach(k => {
        // Propagate mappings from the predecessors.
        let p = k.split('/')
        let pred = p.slice(0, -1).join('/')
        let name = p.pop()

        if (pred.length > 0) {
          pred = this.paths[pred].mapping

          let realPath = [pred, name]
            .filter(s => s)
            .join('/')

          this.paths[k].mapping = realPath
        }

        // Group similar metanodes together.
        let n = name.split('_')
        let prefix = n.slice(0, -1).join('_')
        let suffix = n.pop()

        if (prefix.length > 0) {
          let basePath = [pred, prefix]
            .filter(s => s)
            .join('/')

          if (basePath in this.paths && this.paths[basePath].isMetanode) {
            name = prefix
            this.paths[k].mapping = basePath
          }
        }
      })
    })

    // Metanodes which have no child or only one child are unused.
    let childrenCount = {}
    Object.keys(this.paths).forEach(k => {
      let parent = predecessor(k)

      // FIXME(liautaud)
      if (parent in this.paths &&
          this.paths[parent].mapping in this.paths) {
        parent = this.paths[parent].mapping
      }

      if (!(parent in childrenCount)) {
        childrenCount[parent] = 0
      }

      childrenCount[parent] += 1
    })

    Object.keys(childrenCount).forEach(k => {
      if (childrenCount[k] <= 1) {
        this.paths[k].isUsed = false
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
    if (this.paths[parent].isMetanode && this.paths[parent].isUsed) {
      return this.paths[parent].mapping
    }

    // Otherwise, we reach for the grandparent.
    return this.getParent(parent)
  }

  getPath(index) {
    return this.indexMapping[index]
  }

  getMetanodes() {
    return Object.keys(this.paths)
    .filter(k => this.paths[k].isMetanode)
    .filter(k => this.paths[k].isUsed)
    .map(k => {
      return {
        data: {
          id: this.paths[k].mapping,
          oid: -1,
          parent: this.getParent(k),

          op: 'Meta',
          label: this.paths[k].mapping,
        }
      }
    })
  }
}
