<template>
  <div class="graph-container" ref="container">
    <v-card v-if="highlighted" class="node-infos">
      <v-card-title primary-title>
        <div>
          <div class="headline">{{ highlighted.id() }}</div>
          <span class="grey--text">Operation: {{ highlighted.data('op') }}</span>
        </div>
      </v-card-title>
<!--       <v-card-actions>
        <v-btn flat>Share</v-btn>
        <v-btn flat color="primary">Explore</v-btn>
        <v-spacer></v-spacer>
        <v-btn icon @click.native="show = !show">
          <v-icon>{{ show ? 'keyboard_arrow_down' : 'keyboard_arrow_up' }}</v-icon>
        </v-btn>
      </v-card-actions>
      <v-slide-y-transition>
        <v-card-text v-show="show">
          I'm a thing. But, like most politicians, he promised more than he could deliver. You won't have time for sleeping, soldier, not with all the bed making you'll be doing. Then we'll go with that data file! Hey, you add a one and two zeros to that or we walk! You're going to do his laundry? I've got to find a way to escape.
        </v-card-text>
      </v-slide-y-transition> -->
    </v-card>
    <div class="graph-area" ref="area"></div>
  </div>
</template>

<style scoped>
  .graph-container {
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    right: 0;
    z-index: 1;
  }

  .graph-area {
    width: 100%;
    height: 100%;
    z-index: 1;
  }

  .node-infos {
    position: absolute;
    top: 98px;
    right: 28px;
    z-index: 1000;
    max-width: 500px;
  }
</style>

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
      highlighted: null,
      show: false,// FIXME
    }),

    watch: {
      graph(value) {
        this.redraw()
      },

      highlighted(current, previous) {
        if (previous) {
          previous.removeClass('highlighted')
        }

        if (current) {
          current.addClass('highlighted')
        }
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
          container: this.$refs.area,
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
          .filter('[op = "Meta"]')
          .on('click', e => {
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

        // Prevent selection of expanded metanodes.
        // TODO(liautaud)

        // Handle regular node highlighting.
        this.instance
          .on('click', e => {
            if (e.target === this.instance ||
                !e.target.is('node[op != "Meta"]')) {
              this.highlighted = null
            } else {
              this.highlighted = e.target
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