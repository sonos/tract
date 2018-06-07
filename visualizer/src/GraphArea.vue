<template></template>

<script>
  import cytoscape from 'cytoscape'
  import dagre from 'cytoscape-dagre'
  import klay from 'cytoscape-klay'
  import panzoom from 'cytoscape-panzoom'
  import expand from 'cytoscape-expand-collapse'

  // Register extensions.
  cytoscape.use(dagre)
  cytoscape.use(klay)
  cytoscape.use(panzoom)
  cytoscape.use(expand)

  export default {
    data: () => ({
      instance: null,
    }),

    mounted() {
      this.instance = cytoscape({
        container: this.$el,
        elements: [
          {data: { id: 'a' }},
          {data: { id: 'b' }},
          {data: { id: 'ab', source: 'a', target: 'b' }},
        ],
      })
    }
  }
</script>