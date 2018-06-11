<template></template>

<script>
  import { Hierarchy, stringToColor } from './helpers'
  import { graphStyle } from './styles'

  import jquery from 'jquery'

  import cytoscape from 'cytoscape'
  import dagre from 'cytoscape-dagre'
  import expand from 'cytoscape-expand-collapse'
  import layout from './layout'

  // Register extensions.
  cytoscape.use(layout)
  cytoscape.use(dagre)
  cytoscape.use(expand, jquery)

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
              type: 'node',
              name: hierarchy.getName(n.name),
              parent: hierarchy.getParent(n.name),
              background: 'hsl(' + stringToColor(n.op_name) + ', 100%, 90%)',
              border: 'hsl(' + stringToColor(n.op_name) + ', 40%, 80%)',
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

        this.expand = this.instance.expandCollapse({
          layoutBy: {
            // name: 'dagre',
            name: 'tensorflow',
            // animate: 'end',
            randomize: false,
            fit: false,
            padding: 80,
          },
          fisheye: false,
          undoable: false,
        })

        this.expand.collapseAll()
        this.layout.run()

        this.instance.zoom(1.5)
        this.instance.center()

        // Handle double click on metanodes.
        let clickTimer = null
        this.instance.nodes()
          .filter('[type = "metanode"]')
          .on('click', (e) => {
            if (!clickTimer) {
              clickTimer = setTimeout(() => clickTimer = null, 400)
            } else {
              clickTimer = null

              if (this.expand.isExpandable(e.target)) {
                this.expand.expand(e.target)
              } else {
                this.expand.collapse(e.target)
              }
            }
          })
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