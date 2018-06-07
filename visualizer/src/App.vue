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
        class="mx-5"></v-text-field>

      <v-btn icon
        @click="$refs.input.open()">
        <v-icon>open_in_browser</v-icon>
      </v-btn>
      <v-btn icon class="mr-4">
        <v-icon>settings</v-icon>
      </v-btn>
    </v-toolbar>

    <v-content>
      <graph-area style="width: 100%; height: 100%;"></graph-area>
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

    props: []
  }
</script>