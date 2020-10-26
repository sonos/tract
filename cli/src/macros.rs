#[macro_export]
macro_rules! dispatch_model {
    ($model: expr, $expr: expr) => {
        (|model: &dyn Model| {
            if let Some(m) = model.downcast_ref::<tract_hir::prelude::InferenceModel>() {
                return $expr(m);
            }
            if let Some(m) = model.downcast_ref::<TypedModel>() {
                return $expr(m);
            }
            #[cfg(feature = "pulse")]
            {
                if let Some(m) = model.downcast_ref::<PulsedModel>() {
                    return $expr(m);
                }
            }
            unreachable!()
        })($model)
    };
}

#[macro_export]
macro_rules! dispatch_model_no_pulse {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_ref::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_ref::<TypedModel>() {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    };
}

#[macro_export]
macro_rules! dispatch_model_mut_no_pulse {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_mut::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_mut::<TypedModel>() {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    };
}
