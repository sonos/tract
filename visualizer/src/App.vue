<template>
  <v-app>
    <v-toolbar color="indigo" dark fixed app>
      <v-toolbar-title class="mx-4">
        TFDeploy Visualizer
      </v-toolbar-title>

      <v-text-field
        prepend-icon="search"
        hide-details
        single-line
        class="mx-4"></v-text-field>

      <v-btn icon
        @click="$refs.input.open()"
        class="ml-4">
        <v-icon>open_in_browser</v-icon>
      </v-btn>
      <v-btn icon class="mr-4">
        <v-icon>settings</v-icon>
      </v-btn>
    </v-toolbar>

    <v-content>
      <graph-area style="width: 100%; height: 100%;" :graph="graph"></graph-area>
    </v-content>

    <json-input
      v-model="graph"
      ref="input">
    </json-input>
  </v-app>
</template>

<script>
  import JsonInput from './JsonInput.vue'
  import GraphArea from './GraphArea.vue'

  export default {
    data: () => ({
      graph: null
    }),

    components: {
      JsonInput,
      GraphArea,
    },

    props: [],

    /** Fetches the last graph from the sessionStorage. */
    mounted() {
      let graph = sessionStorage.getItem('tfd-state')
      if (graph) {
        this.graph = JSON.parse(graph)
      }
    },

    watch: {
      /** Stores the graph in the sessionStorage when updated. */
      graph(value) {
        console.log('Storing the graph in the sessionStorage.')
        sessionStorage.setItem('tfd-state', JSON.stringify(value))
      }
    }
  }
</script>