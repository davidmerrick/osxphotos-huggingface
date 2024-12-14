from lib.classify import Classifier
from lib.osxphotos_utils import construct_query_options, PhotoProcessContext, ProcessResult, ProcessResultStatus
from lib.osxphotos_utils.photoprocessor import PhotoProcessor

class PhotoFlagger:
    """
    This class just orchestrates the processing of photos by multiple classifiers.
    """
    def __init__(
        self,
        verbose_mode,
        library_path,
        classifier: list[Classifier],
        keystore_name="multi_flagger.db"
    ):
        self.processor = PhotoProcessor(
            keystore_name=keystore_name,
            verbose_mode=verbose_mode,
            library_path=library_path
        )
        self.classifiers = classifier  # Store multiple classifiers

    def _get_exclude_keywords(self):
        return [f"validated_{classifier.name}" for classifier in self.classifiers]

    def flag_photos(self, dry_run, reset, selected):
        query_options = construct_query_options(selected, exclude_keywords=self._get_exclude_keywords())
        self.processor.process_photos(
            query_options,
            self._process_photo,
            dry_run=dry_run,
            reset=reset
        )

    def _process_photo(self, ctx: PhotoProcessContext) -> ProcessResult:
        if not ctx.photo.path_derivatives:
            ctx.logger.debug(f"Skipping {ctx.original_filename}; could not find photo path")
            return ProcessResult(status=ProcessResultStatus.SKIPPED)

        # Use all classifiers to process the photo
        flags = []
        for classifier in self.classifiers:
            classification = classifier.classify(ctx.photo.path_derivatives[0])
            if classification:
                if isinstance(classification, bool):
                    flags.append(f"flagged_{classifier.name}")
                else:
                    flags.append(f"flagged_{classifier.name}_{classification}")

        if len(flags) > 0:
            ctx.logger.debug(f"Image flagged with keywords: {', '.join(flags)}")
            return ProcessResult(ProcessResultStatus.FLAGGED, flags)
        else:
            ctx.logger.debug("Image was not flagged")
            return ProcessResult(ProcessResultStatus.SKIPPED)
