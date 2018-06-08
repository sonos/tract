<template></template>

<script>
  import { Hierarchy } from './helpers'
  import { graphStyle } from './styles'

  import cytoscape from 'cytoscape'
  // import dagre from 'cytoscape-dagre'
  // import klay from 'cytoscape-klay'
  import layout from './layout'
  // import expand from 'cytoscape-expand-collapse'

  // Register extensions.
  // cytoscape.use(dagre)
  // cytoscape.use(klay)
  cytoscape.use(layout)
  // cytoscape.use(expand)

  export default {
    props: {
      graph: Array
    },

    data: () => ({
      instance: null,
      parsed: null,
    }),

    watch: {
      graph(value) {
        this.redraw()
      }
    },

    methods: {
      redraw() {
        console.log('Redrawing graph with new data.')

        if (!this.graph) {
          return
        }

        if (this.instance) {
          this.instance.destroy()
        }

        let hierarchy = new Hierarchy(this.graph[0])

        let nodes = this.graph[0]
          .map(n => ({
            data: {
              id: n.name,
              op: n.op_name,
              oid: n.id,
              name: hierarchy.getName(n.name),
              parent: hierarchy.getParent(n.name),
            }
          }))

        let metanodes = hierarchy.getMetanodes()

        let edges = this.graph[1]
          .filter(e => e.from_node !== null && e.to_node !== null)
          .map(e => ({
            data: {
              id: 'e' + e.id,
              source: hierarchy.getPath(e.from_node),
              target: hierarchy.getPath(e.to_node),
            }
          }))

        let graph = nodes.concat(metanodes, edges)

        this.instance = cytoscape({
          container: this.$el,
          elements: graph,

          wheelSensitivity: 0.1,
          autoungrabify: true,

          style: graphStyle,
        })

        this.layout = this.instance.layout({
          name: 'tensorflow',
        })

        this.layout.run()
      }
    },

    mounted() {
      this.redraw()
    },

    destroyed() {
      this.instance.destroy()
    }
  }
</script>