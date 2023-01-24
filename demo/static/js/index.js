function getJson(url, callback) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.onload = () => {
    if (xhr.status === 200) {
      callback(xhr.response, xhr.status, xhr.statusText);
    } else {
      callback(xhr.response, xhr.status, xhr.statusText);
    }
  };
  xhr.send();
}

function getQueryParams() {
  if (document.URL.includes("?")) {
    const paramPairs = document.URL.split("?")
        .pop()
        .split("&")
        .map(pair => pair.split("="))
        .map(([k, v]) => [k, decodeURIComponent(v.replace(/\+/g, "%20"))]);

    return Object.assign.apply(null, paramPairs.map(([k, v]) => ({[k]: v})));
  } else {
    return [];
  }
}

function main() {
  const $iconButtonRipple = mdc.ripple.MDCRipple.attachTo(document.querySelector(".mdc-button"));
  $iconButtonRipple.unbounded = true;

  new mdc.textField.MDCTextField(document.querySelector(".mdc-text-field"));

  const params = getQueryParams();

  if (params.q) {
    const $videos = document.getElementById("videos");

    $videos.innerHTML = `
        <div role="progressbar" class="mdc-linear-progress mdc-linear-progress--indeterminate"
            aria-label="Example Progress Bar" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0">
          <div class="mdc-linear-progress__buffer">
            <div class="mdc-linear-progress__buffer-bar"></div>
            <div class="mdc-linear-progress__buffer-dots"></div>
          </div>
          <div class="mdc-linear-progress__bar mdc-linear-progress__primary-bar">
            <span class="mdc-linear-progress__bar-inner"></span>
          </div>
          <div class="mdc-linear-progress__bar mdc-linear-progress__secondary-bar">
            <span class="mdc-linear-progress__bar-inner"></span>
          </div>
        </div>`;

    document.getElementById("q").value = params.q;

    // noinspection HttpUrlsUsage
    getJson(`search?${document.URL.split("?").pop()}`, (response, status, statusText) => {
      if (status === 200) {
        $videos.innerHTML = "";
        response.forEach((result, i) => {
          $videos.innerHTML += `
            <div class="video" id="video-${i}">
              <p class="loading-video">Loading video…</p>
            </div>`;

          // noinspection JSUnresolvedFunction,JSUnresolvedVariable,JSUnusedGlobalSymbols
          new YT.Player(`video-${i}`, {
            videoId: result.video_id,
            playerVars: {
              enablejsapi: 1,
              modestbranding: 1,
              cc_load_policy: 1,
              iv_load_policy: 3,      // Hide the Video Annotations
              rel: 0,
              start: Math.round(result.start_time - 1),
              end: Math.round(result.end_time + 1),
            },
            // FIXME: the player events are only fired for the last video.
            //  I've unsuccessfully tried many things to make it work.
            events: {
              onApiChange: e => e.target.setOption("captions", "track", {languageCode: "en"}),
            },
          });
        });
      } else {
        $videos.innerHTML = `
          <p class="error">
            ${response ? response.message : status + " " + statusText}
          </p>
          <p>
            Make sure the query syntax is valid.
            The query needs to be a
            <a href="https://spacy.io/api/matcher#patterns" target="_blank">
              spaCy <code>Matcher</code> <code>Pattern</code>
            </a>.
            It should be a JSON array composed of JSON objects such as: 
          </p>
          <p>
            <code>[{"LOWER": "this", "DEP": {"IN": ["nsubj", "dobj", "iobj"]}}]</code>
          </p>`;
      }
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, false);
} else {
  main();
}
