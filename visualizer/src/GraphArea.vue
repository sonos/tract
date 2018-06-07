<template></template>

<script>
  import cytoscape from 'cytoscape'
  import dagre from 'cytoscape-dagre'
  import klay from 'cytoscape-klay'
  import expand from 'cytoscape-expand-collapse'

  // Register extensions.
  cytoscape.use(dagre)
  // cytoscape.use(klay)
  // cytoscape.use(expand)

  export default {
    props: {
      graph: Array
    },

    data: () => ({
      instance: null,
    }),

    watch: {
      graph(value) {
        this.redraw()
      }
    },

    methods: {
      redraw() {
        console.log('Redrawing graph with new data.')

        if (!this.instance || !this.graph) {
          return
        }

        let nodes = this.graph[0]
          .map(n => ({
            data: {
              id: n.id,
              name: n.name.split('/').slice(-1)[0],
              opName: n.op_name,
              fullName: n.name,
              width: (n.inputs.length > 0) ? '25px' : '10px',
              height: (n.inputs.length > 0) ? '15px' : '10px',
            }
          }))

        let edges = this.graph[1]
          .filter(e => e.from_node !== null && e.to_node !== null)
          .map(e => ({
            data: {
              id: 'e' + e.id,
              source: e.from_node,
              target: e.to_node,
            }
          }))

        let graph = nodes.concat(edges)

        console.log('Using new graph:', graph)

        this.instance.json({
          elements: graph,
        })

        this.layout = this.instance.layout({
          name: 'dagre',
          nodeSep: 40,
          rankSep: 40,
          // name: 'klay',
        })

        this.layout.run()
      }
    },

    mounted() {
      let cy = cytoscape({
        container: this.$el,
        elements: [],

        wheelSensitivity: 0.1,
        autoungrabify: true,

        style: [
          {
            selector: 'node',
            style: {
              'width': 'data(width)',
              'height': 'data(height)',
              'background-color': '#ddd',
              'border-width': '2px',
              'border-color': '#bbb',
              'shape': 'ellipse',
              'label': 'data(name)',
              'font-size': '9px',
              'text-margin-y': '-8px',
            }
          },

          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#ccc',
              'curve-style': 'bezier',
              'target-arrow-color': '#ccc',
              'target-arrow-shape': 'vee',
              'arrow-scale': '.8',
            }
          }
        ],
      })

      cy.nodes().on('click', function() {
        this.select()
      })

      this.instance = cy
    },

    destroyed() {
      this.instance.destroy()
    }
  }
</script>