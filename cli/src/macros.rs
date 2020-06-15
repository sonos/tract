#[macro_export]
macro_rules! dispatch_model {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_ref::<tract_hir::prelude::InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<PulsedModel>() {
            $expr(m)
        } else {
            unreachable!()
        }
    };
}

#[macro_export]
macro_rules! dispatch_model_no_pulse {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_ref::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    };
}
