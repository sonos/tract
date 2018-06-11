<template></template>

<script>
  import { Hierarchy, stringToColor } from './helpers'
  import { graphStyle } from './styles'

  import jquery from 'jquery'

  import cytoscape from 'cytoscape'
  import expand from 'cytoscape-expand-collapse'
  import layout from './layout'

  // Register extensions.
  cytoscape.use(layout)
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

        // let graph = nodes.concat(edges)
        // DEBUG(liautaud)
        // let graph = [
        //   {data: {id: 'a', op: 'noconst', name: 'a'}},
        //   {data: {id: 'b', op: 'noconst', name: 'b'}},
        //   {data: {id: 'c', op: 'noconst', name: 'c'}},
        //   {data: {id: 'd', op: 'noconst', name: 'd'}},
        //   {data: {id: 'e', op: 'noconst', name: 'e'}},
        //   {data: {id: 'ab', source: 'a', target: 'b'}},
        //   {data: {id: 'ac', source: 'a', target: 'c'}},
        //   {data: {id: 'bd', source: 'b', target: 'd'}},
        //   {data: {id: 'cd', source: 'c', target: 'd'}},
        //   {data: {id: 'de', source: 'd', target: 'e'}},
        //   {data: {id: 'ae', source: 'a', target: 'e'}},
        // ]

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
            name: "tensorflow",
            randomize: false,
            fit: true,
            padding: 80,
          },
          fisheye: false,
          undoable: false,
        })

        this.expand.collapseAll()

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